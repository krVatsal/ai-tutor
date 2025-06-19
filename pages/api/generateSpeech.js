import axios from 'axios';

export default async function handler(req, res) {
  const { conversation_id, text } = req.body;
  
  try {
    const { data } = await axios.post(
      `${process.env.TAVUS_API_URL}/speech`,
      { 
        conversation_id, 
        text 
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': process.env.TAVUS_API_KEY
        }
      }
    );
    
    res.status(200).json(data);
  } catch (err) {
    console.error('Error generating speech:', err.response?.data || err.message);
    res.status(500).json({ 
      error: err.response?.data?.message || err.message 
    });
  }
}