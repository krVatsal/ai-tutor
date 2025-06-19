// Frontend API calls to FastAPI backend

const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://your-api-domain.com' 
  : 'http://localhost:8000';

// Create persona with document context
export async function createPersonaWithDocument(documentText: string, documentName: string) {
  try {
    const response = await fetch(`${API_BASE_URL}/create-persona`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        document_text: documentText,
        document_name: documentName
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
    const response = await fetch(`${API_BASE_URL}/create-conversation`, {
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

// Generate video response
export async function generateVideoResponse(text: string, conversationId: string) {
  try {
    const response = await fetch(`${API_BASE_URL}/generate-speech`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        conversation_id: conversationId,
        text: text
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
    const response = await fetch(`${API_BASE_URL}/speech-status/${speechId}`);
    
    if (!response.ok) {
      throw new Error(`Failed to get speech status: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting video status:', error);
    throw error;
  }
}

// Legacy function for backward compatibility
export async function createConversation() {
  try {
    // Try to get stored persona ID
    const storedPersonaId = localStorage.getItem('currentPersonaId');
    const storedDocumentName = localStorage.getItem('currentDocumentName');
    
    if (storedPersonaId && storedDocumentName) {
      return await createConversationWithPersona(storedPersonaId, storedDocumentName);
    }
    
    throw new Error('No persona found. Please upload a document first.');
  } catch (error) {
    console.error('Error creating conversation:', error);
    throw error;
  }
}