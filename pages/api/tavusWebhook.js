export default async function handler(req, res) {
  const event = req.body;
  
  console.log("Tavus webhook received:", {
    type: event.type,
    conversation_id: event.conversation_id,
    timestamp: new Date().toISOString()
  });
  
  try {
    switch (event.type) {
      case 'conversation.started':
        console.log('Conversation started:', event.conversation_id);
        break;
        
      case 'conversation.ended':
        console.log('Conversation ended:', event.conversation_id);
        break;
        
      case 'utterance':
        console.log('User utterance:', event.utterance);
        // Handle user speech input
        break;
        
      case 'tool_call':
        console.log('Tool call requested:', event.tool_name);
        // Handle tool calls like lookup_doc
        if (event.tool_name === 'lookup_doc') {
          // Implement document lookup logic here
          const query = event.parameters?.query;
          // Return relevant document snippet
        }
        break;
        
      default:
        console.log('Unknown event type:', event.type);
    }
    
    res.status(200).json({ success: true });
  } catch (error) {
    console.error('Webhook processing error:', error);
    res.status(500).json({ error: 'Webhook processing failed' });
  }
}