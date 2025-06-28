"use client";

import { useState, useEffect } from "react";
import { Video, Clock, ExternalLink, Loader2, Play, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { cn } from "@/lib/utils";
import { createConversationWithAuth } from "@/lib/tavus";
import { useAuth } from '@clerk/nextjs';

interface AvatarDisplayProps {
  isProcessing: boolean;
  lastMessage: string;
  videoEnabled: boolean;
  personaId?: string | null;
  documentName?: string | null;
}

interface VideoCallUsage {
  calls_used: number;
  calls_remaining: number;
  total_duration_seconds: number;
  total_duration_remaining_seconds: number;
  max_calls_per_day: number;
  max_duration_per_call: number;
  max_total_duration_per_day: number;
}

export function AvatarDisplay({ 
  isProcessing, 
  lastMessage, 
  videoEnabled, 
  personaId,
  documentName 
}: AvatarDisplayProps) {
  const { getToken } = useAuth();
  const [isGeneratingVideo, setIsGeneratingVideo] = useState(false);
  const [timeRemaining, setTimeRemaining] = useState(1200); // 20 minutes in seconds
  const [videoSessionActive, setVideoSessionActive] = useState(false);
  const [conversationUrl, setConversationUrl] = useState<string | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [videoCallUsage, setVideoCallUsage] = useState<VideoCallUsage | null>(null);
  const [canStartCall, setCanStartCall] = useState(true);
  const [usageError, setUsageError] = useState<string | null>(null);

  // Fetch video call usage on component mount
  useEffect(() => {
    fetchVideoCallUsage();
  }, []);

  const fetchVideoCallUsage = async () => {
    try {
      const token = await getToken();
      if (!token) return;

      const response = await fetch('http://localhost:8000/api/video-call-usage', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setVideoCallUsage(data.usage);
        setCanStartCall(data.can_start_call);
      } else {
        console.error('Failed to fetch video call usage');
      }
    } catch (error) {
      console.error('Error fetching video call usage:', error);
    }
  };

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
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const initiateVideoChat = async () => {
    if (!personaId || !documentName) {
      console.error('Missing persona ID or document name');
      return;
    }

    if (!canStartCall) {
      setUsageError("Daily video call limit reached. You can only make 2 video calls per day with a maximum of 20 minutes each.");
      return;
    }

    try {
      setIsGeneratingVideo(true);
      setUsageError(null);
      
      // Get authentication token
      const token = await getToken();
      if (!token) {
        throw new Error('Authentication token not available');
      }
      
      // Create conversation with the persona
      const response = await createConversationWithAuth(personaId, documentName, token);
      
      if (response.conversation_url && response.conversation_id) {
        setConversationUrl(response.conversation_url);
        setConversationId(response.conversation_id);
        setVideoSessionActive(true);
        setTimeRemaining(1200); // Reset to 20 minutes
        
        // Store conversation details
        localStorage.setItem('currentConversationId', response.conversation_id);
        localStorage.setItem('currentConversationUrl', response.conversation_url);
        
        // Refresh usage data
        await fetchVideoCallUsage();
        
        // Open the conversation URL in a new tab
        window.open(response.conversation_url, '_blank', 'width=800,height=600');
      }
    } catch (error: any) {
      console.error('Error creating video conversation:', error);
      if (error.message?.includes('Daily limit reached')) {
        setUsageError(error.message);
        await fetchVideoCallUsage(); // Refresh usage data
      } else {
        setUsageError('Failed to start video chat. Please try again.');
      }
    } finally {
      setIsGeneratingVideo(false);
    }
  };

  const openVideoChat = () => {
    if (conversationUrl) {
      window.open(conversationUrl, '_blank', 'width=800,height=600');
    }
  };

  const canStartVideo = personaId && documentName && !videoSessionActive && canStartCall;

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
          <div className="w-32 h-32 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
            <Video className="w-16 h-16 text-white" />
          </div>
          {videoSessionActive && (
            <div className="absolute -top-2 -right-2 w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
              <div className="w-3 h-3 bg-white rounded-full animate-pulse" />
            </div>
          )}
        </div>

        {/* Status */}
        <div className="text-center">
          <h3 className="text-lg font-semibold mb-2">
            {videoSessionActive ? "Video Chat Active" : "Mira AI Tutor"}
          </h3>
          <p className="text-sm text-muted-foreground">
            {videoSessionActive 
              ? "Your video conversation is running" 
              : "Start a video conversation to discuss your document"
            }
          </p>
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
            
            {personaId && !videoSessionActive && (
              <div className="text-xs text-muted-foreground">
                AI persona ready with document context
              </div>
            )}
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
              value={(timeRemaining / 3600) * 100} 
              className="h-2"
            />
            <p className="text-xs text-center text-muted-foreground">
              Video session will expire when timer reaches zero
            </p>
          </div>
        )}

        {/* Information */}
        <div className="space-y-2 text-sm text-muted-foreground">
          <div className="flex items-center justify-between">
            <span>Status:</span>
            <Badge variant={videoSessionActive ? "default" : "secondary"}>
              {videoSessionActive ? "Active" : "Ready"}
            </Badge>
          </div>
          
          {videoCallUsage && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span>Daily Calls:</span>
                <span className="font-medium">
                  {videoCallUsage.calls_used}/{videoCallUsage.max_calls_per_day}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span>Daily Duration:</span>
                <span className="font-medium">
                  {Math.floor(videoCallUsage.total_duration_seconds / 60)}/{Math.floor(videoCallUsage.max_total_duration_per_day / 60)} min
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span>Calls Remaining:</span>
                <span className="font-medium text-green-600">
                  {videoCallUsage.calls_remaining}
                </span>
              </div>
            </div>
          )}
          
          {videoSessionActive && (
            <div className="flex items-center justify-between">
              <span>Time Remaining:</span>
              <span className="font-mono font-medium">
                {formatTime(timeRemaining)}
              </span>
            </div>
          )}
        </div>

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
              ) : !canStartCall ? (
                <>
                  <AlertCircle className="mr-2 h-4 w-4" />
                  Daily Limit Reached
                </>
              ) : (
                <>
                  <Video className="mr-2 h-4 w-4" />
                  Start Video Chat with Mira
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
                  <p className="font-medium">1-hour video sessions</p>
                  <p className="text-muted-foreground text-xs">
                    Each video chat session lasts up to 1 hour
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
                      Mira has been trained on your specific document content
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

        {/* Usage Error Alert */}
        {usageError && (
          <Alert className="mt-4">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{usageError}</AlertDescription>
          </Alert>
        )}
      </CardContent>
    </div>
  );
}