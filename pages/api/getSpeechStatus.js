import axios from 'axios';

export default async function handler(req, res) {
  const { speech_id } = req.query;
  
  try {
    const { data } = await axios.get(
      `${process.env.TAVUS_API_URL}/speech/${speech_id}`,
      {
        headers: { 
          'x-api-key': process.env.TAVUS_API_KEY 
        }
      }
    );
    
    res.status(200).json(data);
  } catch (err) {
    console.error('Error getting speech status:', err.response?.data || err.message);
    res.status(500).json({ 
      error: err.response?.data?.message || err.message 
    });
  }
}