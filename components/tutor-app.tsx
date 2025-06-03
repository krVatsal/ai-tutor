"use client";

import { useState } from "react";
import { AppHeader } from "@/components/app-header";
import { DocumentUpload } from "@/components/document-upload";
import { VoiceInput } from "@/components/voice-input";
import { AvatarDisplay } from "@/components/avatar-display";
import { TranscriptDisplay } from "@/components/transcript-display";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileText, FileUp } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Message } from "@/types/chat";
import { cn } from "@/lib/utils";

// API base URL configuration - use relative URL to match protocol
const API_BASE_URL = "http://0.0.0.0:8000/api";

export function TutorApp() {
  const [activeDocument, setActiveDocument] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Hello! I'm Mira, your AI tutor. How can I help you today?",
    },
  ]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [tab, setTab] = useState<string>("chat");
  const [videoEnabled, setVideoEnabled] = useState(true);

  const handleDocumentUpload = async (file: File) => {
    setIsUploading(true);
    
    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to upload document");
      }

      const data = await response.json();
      
      setActiveDocument(file.name);
      
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `I've processed the document "${file.name}". What would you like to know about it?`,
        },
      ]);
      
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

  const handleVoiceInput = async (transcript: string) => {
    if (!transcript.trim()) return;
    
    setMessages((prev) => [
      ...prev,
      { role: "user", content: transcript },
    ]);
    
    setIsProcessing(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          query: transcript,
          document_name: activeDocument
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get AI response");
      }

      const data = await response.json();
      
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.response },
      ]);
      
      generateAvatarResponse(data.response);
      
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

  const generateAvatarResponse = async (text: string) => {
    if (!videoEnabled) return;

    try {
      const response = await fetch(`${API_BASE_URL}/generate-avatar-response`, {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: `text=${encodeURIComponent(text)}`,
      });

      if (!response.ok) {
        throw new Error("Failed to generate avatar response");
      }

      const data = await response.json();
      console.log("Avatar response generated:", data);
    } catch (error) {
      console.error("Error generating avatar response:", error);
    }
  };

  const handleSummarizeDocument = async () => {
    if (!activeDocument) return;
    
    setIsProcessing(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/summarize`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ document_name: activeDocument }),
      });

      if (!response.ok) {
        throw new Error("Failed to summarize document");
      }

      const data = await response.json();
      
      setMessages((prev) => [
        ...prev,
        { role: "user", content: "Can you summarize this document?" },
        { role: "assistant", content: data.summary },
      ]);
      
      generateAvatarResponse(data.summary);
      
    } catch (error) {
      console.error("Error summarizing document:", error);
      
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "I'm sorry, I couldn't summarize the document. Please try again.",
        },
      ]);
    } finally {
      setIsProcessing(false);
    }
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
              "flex-1 flex flex-col space-y-6 md:space-y-8",
              tab !== "chat" && "hidden"
            )}
          >
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 flex-1">
              <Card className="col-span-1 lg:col-span-3">
                <CardContent className="p-4 sm:p-6 h-[500px] flex flex-col">
                  <TranscriptDisplay messages={messages} />
                </CardContent>
              </Card>
              
              <Card className="col-span-1 lg:col-span-2">
                <CardContent className="p-4 sm:p-6 h-[500px] flex flex-col">
                  <AvatarDisplay 
                    isProcessing={isProcessing}
                    lastMessage={messages[messages.length - 1]?.content || ''}
                    videoEnabled={videoEnabled}
                  />
                </CardContent>
              </Card>
            </div>
            
            <VoiceInput 
              onTranscript={handleVoiceInput}
              isProcessing={isProcessing}
            />
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