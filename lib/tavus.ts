import axios from 'axios';

const TAVUS_API_URL = 'https://tavusapi.com/v2';
const TAVUS_API_KEY = process.env.NEXT_PUBLIC_TAVUS_API_KEY;
const TAVUS_PERSONA_ID = 'p296a08ac834';
const TAVUS_REPLICA_ID = 'r9d30b0e55ac';

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

export async function createConversation() {
  try {
    const response = await axios.post(
      `${TAVUS_API_URL}/conversations`,
      {
        replica_id: TAVUS_REPLICA_ID,
        conversation_name: "AI Tutor Chat",
        persona_id: TAVUS_PERSONA_ID
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
    // Check if error is due to conversation limit
    if (error.response?.data?.message?.includes('conversation limit')) {
      // Try to get the last conversation ID from localStorage
      const lastConversationId = localStorage.getItem('lastTavusConversationId');
      if (lastConversationId) {
        const existingConversation = await getExistingConversation(lastConversationId);
        return existingConversation;
      }
    }
    console.error('Error creating conversation:', error);
    throw error;
  }
}

export async function generateVideoResponse(text: string) {
  try {
    // First create a conversation or get existing one
    const conversationResponse = await createConversation();
    
    // Store the conversation ID for future use
    if (conversationResponse.conversation_id) {
      localStorage.setItem('lastTavusConversationId', conversationResponse.conversation_id);
    }
    
    if (conversationResponse.conversation_url) {
      // Open the conversation URL in a new tab
      window.open(conversationResponse.conversation_url, '_blank');
    }

    // Then generate the video response
    const response = await axios.post(
      `${TAVUS_API_URL}/speech`,
      {
        text,
        conversation_id: conversationResponse.conversation_id
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
    console.error('Error generating video response:', error);
    throw error;
  }
}

export async function getVideoStatus(videoId: string) {
  try {
    const response = await axios.get(`${TAVUS_API_URL}/speech/${videoId}`, {
      headers: {
        'x-api-key': TAVUS_API_KEY
      }
    });

    return response.data;
  } catch (error) {
    console.error('Error getting video status:', error);
    throw error;
  }
}