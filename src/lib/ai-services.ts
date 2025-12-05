export interface AIMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  agentType?: string;
}

export interface AIResponse {
  content: string;
  model: string;
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  reasoning?: string;
}

export class ConstructionAIService {
  private static instance: ConstructionAIService;

  public static getInstance(): ConstructionAIService {
    if (!ConstructionAIService.instance) {
      ConstructionAIService.instance = new ConstructionAIService();
    }
    return ConstructionAIService.instance;
  }

  // Placeholder for custom AI model integration
  async processRequest(message: string, context?: any): Promise<AIResponse> {
    // Implement custom AI logic here
    return {
      content: "Custom AI response placeholder.",
      model: "custom-ai-model",
    };
  }
}
