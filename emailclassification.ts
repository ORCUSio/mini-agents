import { HumanMessage } from "@langchain/core/messages";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import {
  Command,
  END,
  GraphNode,
  interrupt,
  MemorySaver,
  START,
  StateGraph,
  StateSchema,
} from "@langchain/langgraph";
import * as z from "zod";

const EmailClassificationSchema = z.object({
  intent: z.enum(["question", "bug", "billing", "feature", "complex"]),
  urgency: z.enum(["low", "medium", "high", "critical"]),
  topic: z.string(),
  summary: z.string(),
});

const EmailAgentState = new StateSchema({
  emailContent: z.string(),
  senderEmail: z.string(),
  emailId: z.string(),

  classification: EmailClassificationSchema.optional(),

  searchResults: z.array(z.string()).optional(),
  customerHistory: z.record(z.string(), z.any()).optional(),
  responseText: z.string().optional(),
});

type EmailClassificationType = z.infer<typeof EmailClassificationSchema>;

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash-lite",
  apiKey: process.env.GEMINI_API_KEY,
});

const readEmail: GraphNode<typeof EmailAgentState> = async (state, config) => {
  console.log(`Processing email: ${state.emailContent}`);
  return {};
};

const classifyIntent: GraphNode<typeof EmailAgentState> = async (
  state,
  config,
) => {
  const structuredLlm = llm.withStructuredOutput(EmailClassificationSchema);

  const classificationPrompt = `
    Analyse this customer email and classify it:
    Email: ${state.emailContent}
    Fromt: ${state.senderEmail}

    Provide classfication includign intent, urgency, topic and summary.
  `;

  const classification = await structuredLlm.invoke(classificationPrompt);

  let nextNdoe:
    | "searchDocumentation"
    | "humanReview"
    | "draftResponse"
    | "bugTracking";

  if (
    classification.intent === "billing" ||
    classification.urgency === "critical"
  ) {
    nextNdoe = "humanReview";
  } else if (
    classification.intent === "question" ||
    classification.intent === "feature"
  ) {
    nextNdoe = "searchDocumentation";
  } else if (classification.intent === "bug") {
    nextNdoe = "bugTracking";
  } else {
    nextNdoe = "draftResponse";
  }

  return new Command({
    update: { classification },
    goto: nextNdoe,
  });
};

const searchDocumentation: GraphNode<typeof EmailAgentState> = async (
  state,
  config,
) => {
  const classification = state.classification!;
  const query = `${classification.intent} ${classification.topic}`;

  let searchResults: string[];

  try {
    searchResults = [
      "Reset password via Settings > Security > Change Password",
      "Password must be atleast 12 characters",
      "Include uppercase, lowercase, numbers, and symbols",
    ];
  } catch (error) {
    searchResults = [`Search temporarily unavilable: ${error}`];
  }
  return new Command({
    update: { searchResults },
    goto: "draftResponse",
  });
};

const bugTracking: GraphNode<typeof EmailAgentState> = async (
  state,
  config,
) => {
  const ticketId = "######";

  return new Command({
    update: { searchResults: [`Bug Ticket ${ticketId} created`] },
    goto: "draftResponse",
  });
};

const draftResponse: GraphNode<typeof EmailAgentState> = async (
  state,
  config,
) => {
  const classification = state.classification;
  const contextSection: string[] = [];

  if (state.searchResults) {
    const formattedDocs = state.searchResults
      .map((doc) => `- ${doc}`)
      .join("\n");
    contextSection.push(`Relevant docs:\n ${formattedDocs}`);
  }
  if (state.customerHistory) {
    contextSection.push(
      `Customer tierL ${state.customerHistory.tier ?? "standard"}`,
    );
  }
  const draftPrompt = `
  Draft a response to this customer email:
  ${state.emailContent}

  Email intent: ${classification!.intent}
  Urgency level: ${classification!.urgency}

  ${contextSection.join("\n\n")}

  Guidelines:
  - Be professional and helpful
  - Address their specific concern
  - Use the provided documentation when relevant
  `;

  const response = await llm.invoke([new HumanMessage(draftPrompt)]);
  const needsReview =
    classification!.urgency === "high" ||
    classification!.urgency === "critical" ||
    classification!.intent === "complex";

  // Route to appropriate next node
  const nextNode = needsReview ? "humanReview" : "sendReply";
  return new Command({
    update: { responseText: response.content.toString() },
    goto: nextNode,
  });
};

const humanReview: GraphNode<typeof EmailAgentState> = async (
  state,
  config,
) => {
  const classfication = state.classification;
  const humanDecision = interrupt({
    emailId: state.emailId,
    originalEmail: state.emailContent,
    draftResponse: state.responseText,
    urgency: classfication?.urgency,
    intent: classfication?.intent,
    action: "Please review and approve/edit this response",
  });
  if (humanDecision.approved) {
    return new Command({
      update: {
        responseText: humanDecision.editResponse || state.responseText,
      },
      goto: "sendReply",
    });
  } else {
    return new Command({ update: {}, goto: END });
  }
};
const sendReply: GraphNode<typeof EmailAgentState> = async (state, config) => {
  console.log(`Sending reply: ${state.responseText!.substring(0, 100)}...`);
  return {};
};

const workflow = new StateGraph(EmailAgentState)
  .addNode("readEmail", readEmail)
  .addNode("classifyIntent", classifyIntent)
  .addNode("searchDocumentation", searchDocumentation, {
    retryPolicy: { maxAttempts: 3 },
  })
  .addNode("bugTracking", bugTracking)
  .addNode("draftResponse", draftResponse)
  .addNode("humanReview", humanReview)
  .addNode("sendReply", sendReply)
  .addEdge(START, "readEmail")
  .addEdge("readEmail", "classifyIntent")
  .addEdge("sendReply", END);

const memory = new MemorySaver();
const app = workflow.compile({ checkpointer: memory });
