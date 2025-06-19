import axios from 'axios';

const TAVUS_API_URL = process.env.NEXT_PUBLIC_TAVUS_API_URL || 'https://tavusapi.com/v2';
const TAVUS_API_KEY = process.env.NEXT_PUBLIC_TAVUS_API_KEY;
const TAVUS_REPLICA_ID = process.env.NEXT_PUBLIC_TAVUS_REPLICA_ID || 'r9d30b0e55ac';

// Create persona with document context
export async function createPersonaWithDocument(documentText: string, documentName: string) {
  try {
    const response = await fetch('/api/createPersona', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        documentText,
        documentName
      })
    });

    if (!response.ok) {
      throw new Error(`Failed to create persona: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error creating persona:', error);
    throw error;
  }
}

// Create conversation with persona
export async function createConversationWithPersona(personaId: string, documentName: string) {
  try {
    const response = await fetch('/api/createConversation', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        persona_id: personaId,
        document_name: documentName
      })
    });

    if (!response.ok) {
      throw new Error(`Failed to create conversation: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error creating conversation:', error);
    throw error;
  }
}

// Legacy function for backward compatibility
export async function createConversation() {
  try {
    // Try to get stored persona ID
    const storedPersonaId = localStorage.getItem('currentPersonaId');
    const storedDocumentName = localStorage.getItem('currentDocumentName');
    
    if (storedPersonaId) {
      return await createConversationWithPersona(storedPersonaId, storedDocumentName || 'Document');
    }
    
    // Fallback to direct conversation creation
    const response = await axios.post(
      `${TAVUS_API_URL}/conversations`,
      {
        replica_id: TAVUS_REPLICA_ID,
        conversation_name: "AI Tutor Chat",
        conversational_context: "Educational discussion about uploaded document"
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': TAVUS_API_KEY
        }
      }
    );

    return response.data;
  } catch (error) {
    console.error('Error creating conversation:', error);
    throw error;
  }
}

// Generate video response
export async function generateVideoResponse(text: string, conversationId?: string) {
  try {
    const response = await fetch('/api/generateSpeech', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        conversation_id: conversationId,
        text
      })
    });

    if (!response.ok) {
      throw new Error(`Failed to generate speech: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error generating video response:', error);
    throw error;
  }
}

// Get video status
export async function getVideoStatus(speechId: string) {
  try {
    const response = await fetch(`/api/getSpeechStatus?speech_id=${speechId}`);
    
    if (!response.ok) {
      throw new Error(`Failed to get speech status: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting video status:', error);
    throw error;
  }
}

// Get existing conversation
export async function getExistingConversation(conversationId: string) {
  try {
    const response = await axios.get(
      `${TAVUS_API_URL}/conversations/${conversationId}`,
      {
        headers: {
          'x-api-key': TAVUS_API_KEY
        }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error getting conversation:', error);
    throw error;
  }
}