"use client";

import { useState, useEffect } from "react";
import { Video, Clock, ExternalLink, Loader2, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { createConversationWithPersona } from "@/lib/tavus";

interface AvatarDisplayProps {
  isProcessing: boolean;
  lastMessage: string;
  videoEnabled: boolean;
  personaId?: string | null;
  documentName?: string | null;
}

export function AvatarDisplay({ 
  isProcessing, 
  lastMessage, 
  videoEnabled, 
  personaId,
  documentName 
}: AvatarDisplayProps) {
  const [isGeneratingVideo, setIsGeneratingVideo] = useState(false);
  const [timeRemaining, setTimeRemaining] = useState(300); // 5 minutes in seconds
  const [videoSessionActive, setVideoSessionActive] = useState(false);
  const [conversationUrl, setConversationUrl] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);

  // Timer countdown
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (videoSessionActive && timeRemaining > 0) {
      interval = setInterval(() => {
        setTimeRemaining((prev) => {
          if (prev <= 1) {
            setVideoSessionActive(false);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [videoSessionActive, timeRemaining]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const initiateVideoChat = async () => {
    if (!personaId || !documentName) {
      console.error('Missing persona ID or document name');
      return;
    }

    try {
      setIsGeneratingVideo(true);
      
      // Create conversation with the persona
      const response = await createConversationWithPersona(personaId, documentName);
      
      if (response.conversation_url && response.conversation_id) {
        setConversationUrl(response.conversation_url);
        setConversationId(response.conversation_id);
        setVideoSessionActive(true);
        setTimeRemaining(300); // Reset to 5 minutes
        
        // Store conversation details
        localStorage.setItem('currentConversationId', response.conversation_id);
        localStorage.setItem('currentConversationUrl', response.conversation_url);
        
        // Open the conversation URL in a new tab
        window.open(response.conversation_url, '_blank', 'width=800,height=600');
      }
    } catch (error) {
      console.error('Error creating video conversation:', error);
    } finally {
      setIsGeneratingVideo(false);
    }
  };

  const openVideoChat = () => {
    if (conversationUrl) {
      window.open(conversationUrl, '_blank', 'width=800,height=600');
    }
  };

  const canStartVideo = personaId && documentName && !videoSessionActive;

  return (
    <div className="flex flex-col h-full">
      <CardHeader className="pb-4">
        <CardTitle className="text-lg flex items-center gap-2">
          <Video className="h-5 w-5 text-primary" />
          Video Chat with Mira
        </CardTitle>
      </CardHeader>
      
      <CardContent className="flex-1 flex flex-col justify-center items-center space-y-6">
        {/* Avatar Placeholder */}
        <div className="relative">
          <div className="w-32 h-32 rounded-full bg-gradient-to-br from-primary/20 to-primary/5 flex items-center justify-center border-4 border-primary/10">
            <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center">
              <span className="text-2xl font-bold text-primary">M</span>
            </div>
          </div>
          
          {/* Status indicator */}
          <div className={cn(
            "absolute bottom-2 right-2 w-6 h-6 rounded-full border-2 border-background flex items-center justify-center",
            videoSessionActive ? "bg-green-500" : personaId ? "bg-blue-500" : "bg-gray-400"
          )}>
            {videoSessionActive && (
              <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
            )}
            {personaId && !videoSessionActive && (
              <Play className="w-3 h-3 text-white" />
            )}
          </div>
        </div>

        {/* Status and Timer */}
        <div className="text-center space-y-2">
          <div className="flex items-center justify-center gap-2">
            <Badge variant={videoSessionActive ? "default" : personaId ? "secondary" : "outline"}>
              {videoSessionActive 
                ? "Video Session Active" 
                : personaId 
                  ? "Ready for Video Chat" 
                  : "Upload Document First"}
            </Badge>
          </div>
          
          {videoSessionActive && (
            <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
              <Clock className="h-4 w-4" />
              <span>Time remaining: {formatTime(timeRemaining)}</span>
            </div>
          )}
        </div>

        {/* Progress bar for active session */}
        {videoSessionActive && (
          <div className="w-full space-y-2">
            <Progress 
              value={(timeRemaining / 300) * 100} 
              className="h-2"
            />
            <p className="text-xs text-center text-muted-foreground">
              Video session will expire when timer reaches zero
            </p>
          </div>
        )}

        {/* Action Buttons */}
        <div className="space-y-3 w-full">
          {!videoSessionActive ? (
            <Button
              onClick={initiateVideoChat}
              disabled={!canStartVideo || isGeneratingVideo || isProcessing}
              className="w-full"
              size="lg"
            >
              {isGeneratingVideo ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Setting up video chat...
                </>
              ) : !personaId ? (
                <>
                  <Video className="mr-2 h-4 w-4" />
                  Upload Document to Enable Video
                </>
              ) : (
                <>
                  <Video className="mr-2 h-4 w-4" />
                  Start Video Chat with Document
                </>
              )}
            </Button>
          ) : (
            <div className="space-y-2 w-full">
              <Button
                onClick={openVideoChat}
                className="w-full"
                size="lg"
              >
                <ExternalLink className="mr-2 h-4 w-4" />
                Open Video Chat
              </Button>
              
              <p className="text-xs text-center text-muted-foreground">
                Video chat opened in new window. Continue your conversation there!
              </p>
            </div>
          )}
        </div>

        {/* Information */}
        <Card className="w-full bg-muted/50">
          <CardContent className="p-4">
            <div className="space-y-2 text-sm">
              <div className="flex items-start gap-2">
                <Clock className="h-4 w-4 mt-0.5 text-muted-foreground shrink-0" />
                <div>
                  <p className="font-medium">5-minute video sessions</p>
                  <p className="text-muted-foreground text-xs">
                    Each video chat session lasts up to 5 minutes
                  </p>
                </div>
              </div>
              
              <div className="flex items-start gap-2">
                <ExternalLink className="h-4 w-4 mt-0.5 text-muted-foreground shrink-0" />
                <div>
                  <p className="font-medium">Opens in new window</p>
                  <p className="text-muted-foreground text-xs">
                    Video chat will open in a separate browser window
                  </p>
                </div>
              </div>

              {personaId && (
                <div className="flex items-start gap-2">
                  <Video className="h-4 w-4 mt-0.5 text-muted-foreground shrink-0" />
                  <div>
                    <p className="font-medium">Document-aware AI</p>
                    <p className="text-muted-foreground text-xs">
                      Mira has been trained on your specific document
                    </p>
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Processing indicator */}
        {isProcessing && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>Mira is thinking...</span>
          </div>
        )}
      </CardContent>
    </div>
  );
}