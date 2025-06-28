"use client"

import { useState, useEffect } from "react";
import { AppHeader } from "@/components/app-header";
import { DocumentUpload } from "@/components/document-upload";
import { ChatInterface } from "@/components/chat-interface";
import { AvatarDisplay } from "@/components/avatar-display";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileText, FileUp, History } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Message } from "@/types/chat";
import { cn } from "@/lib/utils";
import { createConversationWithPersona } from "@/lib/tavus";
import { useUser, useAuth } from '@clerk/nextjs';

// API base URL configuration
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://your-api-domain.com' 
  : 'https://mira-backend-fcdndhgegjdhghf2.centralindia-01.azurewebsites.net';

export function TutorApp() {
  const { isLoaded, isSignedIn, user } = useUser();
  const { getToken } = useAuth();
  const [activeDocument, setActiveDocument] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Hello! I'm Mira, your AI tutor. Upload a document to get started, and I'll help you understand it better.",
    },
  ]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [tab, setTab] = useState<string>("chat");
  const [videoEnabled, setVideoEnabled] = useState(true);
  const [currentPersonaId, setCurrentPersonaId] = useState<string | null>(null);
  const [documentText, setDocumentText] = useState<string>("");
  const [availableDocuments, setAvailableDocuments] = useState<any[]>([]);

  // Load available documents and restore session on component mount
  useEffect(() => {
    if (isSignedIn) {
      loadAvailableDocuments();
      restoreSession();
    }
  }, [isSignedIn]);

  // Save active document to localStorage whenever it changes
  useEffect(() => {
    if (activeDocument && user) {
      localStorage.setItem(`currentDocumentName_${user.id}`, activeDocument);
    }
  }, [activeDocument, user]);

  // Save current persona to localStorage whenever it changes
  useEffect(() => {
    if (currentPersonaId && user) {
      localStorage.setItem(`currentPersonaId_${user.id}`, currentPersonaId);
    }
  }, [currentPersonaId, user]);

  // Show loading state while Clerk is loading
  if (!isLoaded) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary"></div>
      </div>
    );
  }

  // If user is not signed in, show loading (middleware will handle redirect)
  if (!isSignedIn) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary"></div>
      </div>
    );
  }

  const loadAvailableDocuments = async () => {
    try {
      const token = await getToken();
      console.log('Token for documents:', token ? 'Token exists' : 'No token');
      
      const response = await fetch(`${API_BASE_URL}/api/documents`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        }
      });
      if (response.ok) {
        const data = await response.json();
        // Ensure we have an array of documents
        if (data && Array.isArray(data.documents)) {
          setAvailableDocuments(data.documents);
        } else {
          console.warn('Documents API returned unexpected format:', data)
          setAvailableDocuments([]);
        }
      } else {
        console.error('Failed to load documents:', response.status);
        setAvailableDocuments([]);
      }    } catch (error) {
      console.error('Error loading documents:', error);
      setAvailableDocuments([]);
    }
  };

  const restoreSession = async () => {
    if (!user) return;
    
    // Try to restore previous session from localStorage with user-specific keys
    const storedPersonaId = localStorage.getItem(`currentPersonaId_${user.id}`);
    const storedDocumentName = localStorage.getItem(`currentDocumentName_${user.id}`);
    
    if (storedPersonaId && storedDocumentName) {
      setCurrentPersonaId(storedPersonaId);
      setActiveDocument(storedDocumentName);
        // Load chat history
      try {
        const response = await fetch(`${API_BASE_URL}/api/chat-history/${storedDocumentName}`, {
          headers: {
            'Authorization': `Bearer ${await getToken()}`,
          }
        });
        if (response.ok) {
          const data = await response.json();
          if (data.messages && data.messages.length > 0) {
            const chatMessages = data.messages.map((msg: any) => ({
              role: msg.role,
              content: msg.content
            }));
            setMessages([
              {
                role: "assistant",
                content: `Welcome back, ${user.firstName || 'there'}! I've restored your session with "${storedDocumentName}". Let's continue our discussion.`,
              },
              ...chatMessages
            ]);
          }
        }
      } catch (error) {
        console.error("Error loading chat history:", error);
      }
    }
  };

  const handleDocumentUpload = async (file: File) => {
    if (!user) return;
    
    setIsUploading(true);
    
    try {
      // Upload and process document with FastAPI
      const formData = new FormData();
      formData.append("file", file);

      const token = await getToken();
      console.log('Token for upload:', token ? 'Token exists' : 'No token');
      
      const uploadResponse = await fetch(`${API_BASE_URL}/api/upload`, {
        method: "POST",
        headers: {
          'Authorization': `Bearer ${token}`,
        },
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error("Failed to upload document");
      }

      const uploadData = await uploadResponse.json();
      setActiveDocument(file.name);
      
      // Check if persona was created successfully
      if (uploadData.persona_id) {
        setCurrentPersonaId(uploadData.persona_id);
        
        // Store for later use with user-specific keys
        localStorage.setItem(`currentPersonaId_${user.id}`, uploadData.persona_id);
        localStorage.setItem(`currentDocumentName_${user.id}`, file.name);
        
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Perfect, ${user.firstName || 'there'}! I've processed "${file.name}" and created your personalized AI tutor experience. 

✅ Document processed and vectorized
✅ AI persona "Mira" created with document context
✅ Video chat capability enabled

I'm now ready to discuss the document content through both text and video chat. You can start a video conversation using the "Start Video Chat with Mira" button on the right. What would you like to know about the document?`,
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `I've processed "${file.name}" successfully! 

✅ Document processed and vectorized
⚠️ Video chat setup encountered an issue

I'm ready to help you understand the document through our text chat. What would you like to know about it?`,
          },
        ]);
      }
      
      // Refresh available documents list
      await loadAvailableDocuments();
      setTab("chat");
    } catch (error) {
      console.error("Error uploading document:", error);
      
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "I'm sorry, there was an error processing your document. Please try again.",
        },
      ]);
    } finally {
      setIsUploading(false);
    }
  };

  const handleSendMessage = async (messageContent: string) => {
    if (!messageContent.trim() || !user) return;
    
    setMessages((prev) => [
      ...prev,
      { role: "user", content: messageContent },
    ]);
    
    setIsProcessing(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",        headers: {
          "Content-Type": "application/json",
          'Authorization': `Bearer ${await getToken()}`,
        },
        body: JSON.stringify({ 
          query: messageContent,
          document_name: activeDocument
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get AI response");
      }      const data = await response.json();
      
      // Handle case where response might be an object with result property
      const responseText = typeof data.response === 'object' && data.response?.result 
        ? data.response.result 
        : typeof data.response === 'string' 
          ? data.response 
          : JSON.stringify(data.response);
      
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: responseText },
      ]);
      
    } catch (error) {
      console.error("Error getting AI response:", error);
      
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "I'm sorry, I'm having trouble understanding. Could you try asking again?",
        },
      ]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleVoiceMessage = async (transcript: string) => {
    await handleSendMessage(transcript);
  };

  const handleSummarizeDocument = async () => {
    if (!activeDocument) return;
    await handleSendMessage("Can you summarize this document for me?");
  };
  
  const handleLoadDocument = async (document: any) => {
    if (!user) return;
    
    try {
      setActiveDocument(document.filename);
      localStorage.setItem(`currentDocumentName_${user.id}`, document.filename);
      
      const token = await getToken();
      
      // Try to find existing persona for this document
      const personasResponse = await fetch(`${API_BASE_URL}/api/personas`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        }
      });
      if (personasResponse.ok) {
        const personasData = await personasResponse.json();
        const documentPersona = personasData.personas.find(
          (p: any) => p.document_name === document.filename
        );
        
        if (documentPersona) {
          setCurrentPersonaId(documentPersona.persona_id);
          localStorage.setItem(`currentPersonaId_${user.id}`, documentPersona.persona_id);
        }
      }        // Load chat history
      const historyResponse = await fetch(`${API_BASE_URL}/api/chat-history/${document.filename}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
        }
      });
      if (historyResponse.ok) {
        const historyData = await historyResponse.json();
        if (historyData.messages && historyData.messages.length > 0) {
          const chatMessages = historyData.messages.map((msg: any) => ({
            role: msg.role,
            content: msg.content
          }));
          setMessages([
            {
              role: "assistant",
              content: `I've loaded "${document.filename}" and our previous conversation. How can I help you today, ${user.firstName || 'there'}?`,
            },
            ...chatMessages
          ]);
        } else {
          setMessages([
            {
              role: "assistant",
              content: `I've loaded "${document.filename}". What would you like to know about it?`,
            }
          ]);
        }
      }
      
      setTab("chat");
    } catch (error) {
      console.error("Error loading document:", error);
    }
  };

  return (
    <div className="flex flex-col min-h-screen">
      <AppHeader />
      
      <div className="container mx-auto px-4 py-6 flex-1 flex flex-col">
        <div className="mb-4">
          <h2 className="text-2xl font-bold text-foreground">
            Welcome back, {user.firstName || user.emailAddresses[0]?.emailAddress || 'there'}!
          </h2>
          <p className="text-muted-foreground">Ready to continue learning with Mira?</p>
        </div>
        
        <Tabs 
          value={tab} 
          onValueChange={setTab}
          className="flex-1 flex flex-col"
        >
          <div className="flex justify-between items-center mb-6">
            <TabsList>
              <TabsTrigger value="chat" className="text-base">Chat with Mira</TabsTrigger>
              <TabsTrigger value="upload" className="text-base">Upload Document</TabsTrigger>
              <TabsTrigger value="history" className="text-base">Document History</TabsTrigger>
            </TabsList>
            
            {activeDocument && (
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">
                  Active: {activeDocument}
                </span>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={handleSummarizeDocument}
                  disabled={isProcessing}
                >
                  <FileText className="h-4 w-4 mr-2" />
                  Summarize
                </Button>
              </div>
            )}
          </div>
          
          <TabsContent 
            value="chat" 
            className={cn(
              "flex-1 flex flex-col space-y-6",
              tab !== "chat" && "hidden"
            )}
          >
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1">
              {/* Chat Interface */}
              <Card className="col-span-1 lg:col-span-2">
                <CardContent className="p-0 h-[600px]">
                  <ChatInterface
                    messages={messages}
                    onSendMessage={handleSendMessage}
                    onVoiceMessage={handleVoiceMessage}
                    isProcessing={isProcessing}
                    disabled={!activeDocument}
                  />
                </CardContent>
              </Card>
              
              {/* Avatar Display */}
              <Card className="col-span-1">
                <CardContent className="p-4 h-[600px] flex flex-col">
                  <AvatarDisplay 
                    isProcessing={isProcessing}
                    lastMessage={messages[messages.length - 1]?.content || ''}
                    videoEnabled={videoEnabled}
                    personaId={currentPersonaId}
                    documentName={activeDocument}
                  />
                </CardContent>
              </Card>
            </div>
            
            {!activeDocument && (
              <Card className="p-6 text-center bg-muted/50">
                <FileUp className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                <h3 className="text-lg font-semibold mb-2">No Document Uploaded</h3>
                <p className="text-muted-foreground mb-4">
                  Upload a PDF document to start chatting with Mira about its contents.
                </p>
                <Button onClick={() => setTab("upload")}>
                  Upload Document
                </Button>
              </Card>
            )}
          </TabsContent>
          
          <TabsContent 
            value="upload"
            className={cn(
              "flex-1",
              tab !== "upload" && "hidden"
            )}
          >
            <DocumentUpload 
              onUpload={handleDocumentUpload}
              isUploading={isUploading}
            />
          </TabsContent>
          
          <TabsContent 
            value="history"
            className={cn(
              "flex-1",
              tab !== "history" && "hidden"
            )}
          >
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center gap-2 mb-6">
                  <History className="h-5 w-5" />
                  <h3 className="text-lg font-semibold">Your Document History</h3>
                </div>
                  {(!availableDocuments || availableDocuments.length === 0) ? (
                  <div className="text-center py-8">
                    <FileText className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                    <p className="text-muted-foreground">No documents uploaded yet</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {Array.isArray(availableDocuments) && availableDocuments.map((doc) => (
                      <div 
                        key={doc.id}
                        className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted/50 transition-colors"
                      >
                        <div className="flex items-center gap-3">
                          <FileText className="h-5 w-5 text-red-500" />
                          <div>
                            <p className="font-medium">{doc.filename}</p>
                            <p className="text-sm text-muted-foreground">
                              Uploaded: {new Date(doc.upload_date).toLocaleDateString()}
                            </p>
                          </div>
                        </div>
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => handleLoadDocument(doc)}
                        >
                          Load Document
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}