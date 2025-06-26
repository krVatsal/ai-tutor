// Frontend API calls to FastAPI backend with Clerk authentication

import { useAuth } from '@clerk/nextjs';

const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://your-api-domain.com' 
  : 'http://localhost:8000';

// Helper function to get auth headers
async function getAuthHeaders() {
  // This will be called from components that have access to useAuth
  // For now, we'll get the token from localStorage or context
  const token = localStorage.getItem('clerk-token');
  return {
    'Content-Type': 'application/json',
    'Authorization': token ? `Bearer ${token}` : '',
  };
}

// Create persona with document context
export async function createPersonaWithDocument(documentChunks: string[], documentName: string) {
  try {
    const headers = await getAuthHeaders();
    
    const response = await fetch(`${API_BASE_URL}/create_persona`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        docs: documentChunks,
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
    const headers = await getAuthHeaders();
    
    const response = await fetch(`${API_BASE_URL}/create_conversation`, {
      method: 'POST',
      headers,
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
    const headers = await getAuthHeaders();
    
    const response = await fetch(`${API_BASE_URL}/generate-speech`, {
      method: 'POST',
      headers,
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
    const headers = await getAuthHeaders();
    
    const response = await fetch(`${API_BASE_URL}/speech-status/${speechId}`, {
      headers
    });
    
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

// Enhanced function to create persona with authenticated request
export async function createPersonaWithAuth(documentText: string, documentName: string, token: string) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/create_persona`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({
        docs: [documentText], // Convert single text to array
        document_name: documentName
      })
    });

    if (!response.ok) {
      throw new Error(`Failed to create persona: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error creating persona with auth:', error);
    throw error;
  }
}

// Enhanced function to create conversation with authenticated request
export async function createConversationWithAuth(personaId: string, documentName: string, token: string) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/create_conversation`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
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
    console.error('Error creating conversation with auth:', error);
    throw error;
  }
}