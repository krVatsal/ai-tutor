import axios from 'axios';

const TAVUS_API_URL = 'https://api.tavus.io/v2';

export async function generateVideoResponse(text: string) {
  try {
    const response = await axios.post(`${TAVUS_API_URL}/speech`, {
      text,
      voice_id: process.env.NEXT_PUBLIC_TAVUS_VOICE_ID,
    }, {
      headers: {
        'Authorization': `Bearer ${process.env.NEXT_PUBLIC_TAVUS_API_KEY}`,
        'Content-Type': 'application/json',
      }
    });

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
        'Authorization': `Bearer ${process.env.NEXT_PUBLIC_TAVUS_API_KEY}`,
      }
    });

    return response.data;
  } catch (error) {
    console.error('Error getting video status:', error);
    throw error;
  }
}