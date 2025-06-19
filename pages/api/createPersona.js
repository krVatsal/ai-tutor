import axios from 'axios';

// Helper function to extract and chunk documents (placeholder)
async function extractAndChunkDocs(docs) {
  // This would integrate with your existing document processing
  // For now, return a placeholder
  return ["Document content chunks will be processed here"];
}

export default async function handler(req, res) {
  try {
    const { docs, documentText } = req.body;
    
    // Use provided document text or extract from docs
    const textChunks = documentText ? [documentText] : await extractAndChunkDocs(docs);
    
    const persona = await axios.post(
      `${process.env.TAVUS_API_URL}/personas`,
      {
        persona_name: "Mira - AI Document Tutor",
        system_prompt: `You are Mira, an expert AI tutor specializing in helping students understand documents. You have access to the following document content: ${textChunks.join("\n\n")}. 

Your role is to:
- Help students understand the document content
- Answer questions about the material
- Provide explanations and clarifications
- Break down complex concepts into simpler terms
- Engage in educational discussions about the document

Always be helpful, patient, and encouraging in your responses.`,
        pipeline_mode: "full",
        context: textChunks.join("\n\n"),
        default_replica_id: process.env.TAVUS_REPLICA_ID,
        layers: {
          llm: {
            model: "tavus-llama-3-8b-instruct",
            tools: [
              {
                type: "function",
                function: {
                  name: "lookup_doc",
                  description: "Search and retrieve specific information from the document",
                  parameters: {
                    type: "object",
                    properties: { 
                      query: { 
                        type: "string",
                        description: "The search query to find relevant document content"
                      }
                    },
                    required: ["query"]
                  }
                }
              }
            ]
          },
          tts: {
            tts_engine: "cartesia",
            voice_id: process.env.TAVUS_VOICE_ID || "default"
          },
          perception: { 
            perception_model: "raven-0" 
          }
        }
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': process.env.TAVUS_API_KEY
        }
      }
    );
    
    res.status(200).json(persona.data);
  } catch (err) {
    console.error('Error creating persona:', err.response?.data || err.message);
    res.status(500).json({ 
      error: err.response?.data?.message || err.message,
      details: err.response?.data 
    });
  }
}