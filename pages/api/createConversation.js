import axios from 'axios';

export default async function handler(req, res) {
  try {
    const { persona_id, document_name } = req.body;
    
    const response = await axios.post(
      `${process.env.TAVUS_API_URL}/conversations`,
      {
        replica_id: process.env.TAVUS_REPLICA_ID,
        persona_id: persona_id,
        conversation_name: `Document Discussion: ${document_name || 'Untitled'}`,
        conversational_context: `The user has uploaded a document and wants to discuss it. Help them understand the content, answer questions, and provide educational guidance about the material.`,
        callback_url: `${process.env.NEXTAUTH_URL || process.env.VERCEL_URL || 'http://localhost:3000'}/api/tavusWebhook`,
        properties: {
          max_call_duration: 300, // 5 minutes
          participant_left_timeout: 60,
          language: "english",
          enable_recording: false
        }
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': process.env.TAVUS_API_KEY
        }
      }
    );
    
    res.status(200).json(response.data);
  } catch (err) {
    console.error('Error creating conversation:', err.response?.data || err.message);
    res.status(500).json({ 
      error: err.response?.data?.message || err.message,
      details: err.response?.data 
    });
  }
}