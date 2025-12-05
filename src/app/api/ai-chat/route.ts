import { NextRequest, NextResponse } from 'next/server';
import ConstructionAIService from '@/lib/ai-services';

export async function POST(request: NextRequest) {
  try {
    const { message, agentType, context, userId } = await request.json();

    if (!message) {
      return NextResponse.json(
        { error: 'Message is required' },
        { status: 400 }
      );
    }

    const aiService = ConstructionAIService.getInstance();

    // Check if AI services are configured
    const serviceStatus = aiService.isConfigured();

    let response;

    try {
      // Use the multi-agent conversation handler
      response = await aiService.handleMultiAgentConversation(
        [{ role: 'user', content: message, agentType }],
        agentType || 'suna',
        context
      );
    } catch (error) {
      console.error('AI service error:', error);

      // Return a fallback response with service status info
      return NextResponse.json({
        content: `I'm currently operating in offline mode. ${
          !serviceStatus.openai && !serviceStatus.google
            ? 'Please configure your AI API keys to enable full AI capabilities.'
            : 'Experiencing temporary issues with AI services.'
        }`,
        model: 'fallback-mode',
        serviceStatus,
        timestamp: new Date().toISOString(),
        agentType: agentType || 'suna'
      });
    }

    return NextResponse.json({
      ...response,
      serviceStatus,
      timestamp: new Date().toISOString(),
      agentType: agentType || 'suna'
    });

  } catch (error) {
    console.error('AI chat API error:', error);
    return NextResponse.json(
      {
        error: 'Internal server error',
        content: 'I apologize, but I encountered an error processing your request. Please try again.',
        model: 'error-fallback',
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    );
  }
}
