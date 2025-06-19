"use client"

import { useState } from "react";
import { AppHeader } from "@/components/app-header";
import { DocumentUpload } from "@/components/document-upload";
import { ChatInterface } from "@/components/chat-interface";
import { AvatarDisplay } from "@/components/avatar-display";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileText, FileUp } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Message } from "@/types/chat";
import { cn } from "@/lib/utils";

// API base URL configuration
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://your-api-domain.com' 
  : 'http://localhost:8000';

export function TutorApp() {
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

  const handleDocumentUpload = async (file: File) => {
    setIsUploading(true);
    
    try {
      // Upload and process document with FastAPI
      const formData = new FormData();
      formData.append("file", file);

      const uploadResponse = await fetch(`${API_BASE_URL}/api/upload`, {
        method: "POST",
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
        
        // Store for later use
        localStorage.setItem('currentPersonaId', uploadData.persona_id);
        localStorage.setItem('currentDocumentName', file.name);
        
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Perfect! I've processed "${file.name}" and created a personalized learning experience for you. I'm now ready to discuss the document content through both text and video chat. What would you like to know about it?`,
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `I've processed "${file.name}" successfully! While I couldn't set up the advanced video features, I'm ready to help you understand the document through our text chat. What would you like to know?`,
          },
        ]);
      }
      
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
    if (!messageContent.trim()) return;
    
    setMessages((prev) => [
      ...prev,
      { role: "user", content: messageContent },
    ]);
    
    setIsProcessing(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
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

  return (
    <div className="flex flex-col min-h-screen">
      <AppHeader />
      
      <div className="container mx-auto px-4 py-6 flex-1 flex flex-col">
        <Tabs 
          value={tab} 
          onValueChange={setTab}
          className="flex-1 flex flex-col"
        >
          <div className="flex justify-between items-center mb-6">
            <TabsList>
              <TabsTrigger value="chat" className="text-base">Chat with Mira</TabsTrigger>
              <TabsTrigger value="upload" className="text-base">Upload Document</TabsTrigger>
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
        </Tabs>
      </div>
    </div>
  );
}