import axios from 'axios';

const TAVUS_API_URL = 'https://tavusapi.com/v2';
const TAVUS_API_KEY = process.env.NEXT_PUBLIC_TAVUS_API_KEY;
const TAVUS_PERSONA_ID = 'p296a08ac834';
const TAVUS_REPLICA_ID = 'r9d30b0e55ac';

export async function generateVideoResponse(text: string) {
  try {
    // First create a conversation
    const conversationResponse = await axios.post(
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

    const conversationId = conversationResponse.data.conversation_id;

    // Then generate the video response
    const response = await axios.post(
      `${TAVUS_API_URL}/speech`,
      {
        text,
        conversation_id: conversationId
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