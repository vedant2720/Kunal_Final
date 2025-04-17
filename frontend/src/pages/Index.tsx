import React, { useEffect } from "react";
import { DeepfakeProvider, useDeepfake } from "@/context/DeepfakeContext";
import Header from "@/components/Header";
import Footer from "@/components/Footer";
import UploadArea from "@/components/UploadArea";
import ResultDisplay from "@/components/ResultDisplay";
import { Button } from "@/components/ui/button";
import { checkApiStatus } from "@/services/api";
import { useIsMobile } from "@/hooks/use-mobile";
import { toast } from "@/components/ui/use-toast";

const ApiStatusIndicator = () => {
  const { isApiConnected, setIsApiConnected } = useDeepfake();
  const [isChecking, setIsChecking] = React.useState(false);

  const checkConnection = async () => {
    setIsChecking(true);
    try {
      const isConnected = await checkApiStatus();
      setIsApiConnected(isConnected);
      if (!isConnected) {
        toast({
          variant: "destructive",
          title: "Connection Error",
          description: "Could not connect to the API server. Please make sure it's running.",
        });
      }
    } catch (error) {
      setIsApiConnected(false);
      toast({
        variant: "destructive",
        title: "Connection Error",
        description: "Could not connect to the API server. Please make sure it's running.",
      });
    } finally {
      setIsChecking(false);
    }
  };

  useEffect(() => {
    checkConnection();
    
    // Check connection every 30 seconds
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex items-center space-x-2 text-sm">
      <div className="flex items-center space-x-1.5">
        <div
          className={`w-2.5 h-2.5 rounded-full ${
            isApiConnected ? "bg-green-500" : "bg-red-500"
          } ${isChecking ? "animate-pulse-subtle" : ""}`}
        />
        <span className="text-muted-foreground">
          API Server: {isApiConnected ? "Connected" : "Disconnected"}
        </span>
      </div>
      <Button
        variant="ghost"
        size="sm"
        className="h-7 px-2 text-xs"
        onClick={checkConnection}
        disabled={isChecking}
      >
        {isChecking ? "Checking..." : "Check"}
      </Button>
    </div>
  );
};

const HeroSection = () => {
  const isMobile = useIsMobile();
  
  return (
    <section className="w-full py-12 md:py-24 section-transition hero-background">
      <div className="container px-4 md:px-6">
        <div className="flex flex-col items-center space-y-4 text-center">
          <div className="space-y-2">
            <div className="inline-block rounded-lg bg-primary/10 px-3 py-1 text-sm text-primary">
              Advanced Image Analysis
            </div>
            <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl">
              Detect DeepFakes with AI
            </h1>
            <p className="mx-auto max-w-[700px] text-muted-foreground md:text-xl">
              Our cutting-edge computer vision model can detect manipulated images with high accuracy
            </p>
          </div>
          <div className="w-full max-w-sm space-x-4">
            <ApiStatusIndicator />
          </div>
        </div>
      </div>
    </section>
  );
};

const FeatureSection = () => {
  return (
    <section className="w-full py-12 md:py-24 bg-muted/50 section-transition" id="how-it-works">
      <div className="container px-4 md:px-6">
        <div className="flex flex-col items-center justify-center space-y-4 text-center">
          <div className="space-y-2">
            <div className="inline-block rounded-lg bg-primary/10 px-3 py-1 text-sm text-primary">
              How It Works
            </div>
            <h2 className="text-3xl font-bold tracking-tighter md:text-4xl">
              Powered by Vision Transformers
            </h2>
            <p className="mx-auto max-w-[700px] text-muted-foreground md:text-xl">
              Our application uses state-of-the-art deep learning models to detect image manipulations
            </p>
          </div>
        </div>
        <div className="mx-auto grid max-w-5xl grid-cols-1 gap-6 md:grid-cols-3 lg:gap-12 mt-12">
          {/* Feature 1 */}
          <div className="flex flex-col items-center space-y-4 glass-card p-6">
            <div className="w-16 h-16 rounded-full flex items-center justify-center bg-primary/10">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="text-primary"
              >
                <path d="M2 12a5 5 0 0 0 5 5 8 8 0 0 1 5 2 8 8 0 0 1 5-2 5 5 0 0 0 5-5V7h-5a8 8 0 0 0-5 2 8 8 0 0 0-5-2H2Z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold">Deep Learning</h3>
            <p className="text-sm text-muted-foreground text-center">
              Trained on thousands of images to detect subtle manipulation patterns
            </p>
          </div>
          
          {/* Feature 2 */}
          <div className="flex flex-col items-center space-y-4 glass-card p-6">
            <div className="w-16 h-16 rounded-full flex items-center justify-center bg-primary/10">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="text-primary"
              >
                <path d="M20 16V7a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v9m16 0H4m16 0 1.28 2.55a1 1 0 0 1-.9 1.45H3.62a1 1 0 0 1-.9-1.45L4 16" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold">ViT Model</h3>
            <p className="text-sm text-muted-foreground text-center">
              Vision Transformer architecture for advanced image analysis and pattern recognition
            </p>
          </div>
          
          {/* Feature 3 */}
          <div className="flex flex-col items-center space-y-4 glass-card p-6">
            <div className="w-16 h-16 rounded-full flex items-center justify-center bg-primary/10">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="text-primary"
              >
                <path d="M6 18h8" />
                <path d="M3 22h18" />
                <path d="M14 22a7 7 0 1 0 0-14h-1" />
                <path d="M9 14h2" />
                <path d="M9 12a2 2 0 0 1 2-2h1.5" />
                <path d="M5.2 9h9.6" />
                <path d="M5 6h9" />
                <path d="M8 3h6" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold">Flask API</h3>
            <p className="text-sm text-muted-foreground text-center">
              Optimized Python backend for fast and reliable image processing
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};

const AboutSection = () => {
  return (
    <section className="w-full py-12 md:py-24 section-transition" id="about">
      <div className="container px-4 md:px-6">
        <div className="grid gap-6 lg:grid-cols-2 lg:gap-12">
          <div className="flex flex-col justify-center space-y-4">
            <div className="space-y-2">
              <div className="inline-block rounded-lg bg-primary/10 px-3 py-1 text-sm text-primary">
                About the Project
              </div>
              <h2 className="text-3xl font-bold tracking-tighter md:text-4xl">
                Fighting Misinformation
              </h2>
              <p className="max-w-[600px] text-muted-foreground md:text-xl">
                DeepFakes represent a growing challenge to media authenticity and trust. Our tool helps users identify manipulated content to combat the spread of misinformation.
              </p>
            </div>
            <div className="space-y-2">
              <p className="text-muted-foreground">
                This application combines:
              </p>
              <ul className="ml-6 list-disc text-muted-foreground space-y-2">
                <li>React frontend with elegant design and smooth interactions</li>
                <li>Flask backend API for handling image processing</li>
                <li>Pretrained Vision Transformer model for accurate detection</li>
                <li>State-of-the-art computer vision techniques</li>
              </ul>
            </div>
          </div>
          <div className="flex items-center justify-center">
            <div className="glass-card p-6 w-full max-w-md">
              <h3 className="text-xl font-semibold mb-4">Model Technical Details</h3>
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Architecture</span>
                  <span className="font-medium">Vision Transformer (ViT)</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Accuracy</span>
                  <span className="font-medium">~76% on test dataset</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Processing Time</span>
                  <span className="font-medium">~1.2 seconds per image</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Last Updated</span>
                  <span className="font-medium">{new Date().toLocaleDateString()}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

const DeepfakeDetectorApp = () => {
  return (
    <div className="w-full min-h-screen flex flex-col page-transition">
      <Header />
      <main className="flex-1">
        <HeroSection />
        
        <section className="w-full py-12 md:py-24 section-transition">
          <div className="container px-4 md:px-6">
            <div className="mx-auto grid items-center gap-6 lg:grid-cols-2 lg:gap-12">
              <div className="flex flex-col justify-center space-y-4">
                <div className="space-y-2">
                  <h2 className="text-3xl font-bold tracking-tighter md:text-4xl">
                    Upload an Image to Analyze
                  </h2>
                  <p className="text-muted-foreground">
                    Our AI can detect manipulations in faces, backgrounds, and other image elements with high precision.
                  </p>
                </div>
                <div className="space-y-2">
                  <p className="text-sm text-muted-foreground">
                    Supported file types: JPEG, PNG, WebP
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Maximum file size: 10MB
                  </p>
                </div>
              </div>
              
              <div>
                <UploadArea />
                <ResultDisplay />
              </div>
            </div>
          </div>
        </section>
        
        <FeatureSection />
        <AboutSection />
      </main>
      <Footer />
    </div>
  );
};

const Index = () => (
  <DeepfakeProvider>
    <DeepfakeDetectorApp />
  </DeepfakeProvider>
);

export default Index;
