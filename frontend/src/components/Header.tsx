
import React from "react";
import { cn } from "@/lib/utils";

interface HeaderProps {
  className?: string;
}

const Header: React.FC<HeaderProps> = ({ className }) => {
  return (
    <header className={cn("w-full py-6 animate-fade-in", className)}>
      <div className="container px-4 md:px-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="w-6 h-6 text-primary"
              >
                <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
                <circle cx="12" cy="12" r="3" />
              </svg>
            </div>
            <span className="text-xl font-semibold tracking-tight">DeepFake Detector</span>
          </div>
          <nav className="hidden md:flex items-center space-x-8">
            <a 
              href="#" 
              className="text-sm font-medium hover-scale hover:text-primary transition-colors"
            >
              Home
            </a>
            <a 
              href="#about" 
              className="text-sm font-medium hover-scale hover:text-primary transition-colors"
            >
              About
            </a>
            <a 
              href="#how-it-works" 
              className="text-sm font-medium hover-scale hover:text-primary transition-colors"
            >
              How It Works
            </a>
          </nav>
          <div className="flex items-center">
            <a 
              href="https://github.com/your-username/deepfake-detector" 
              target="_blank" 
              rel="noopener noreferrer"
              className="p-2 rounded-full hover:bg-secondary transition-colors"
              aria-label="GitHub Repository"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="text-foreground/80"
              >
                <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
                <path d="M9 18c-4.51 2-5-2-7-2" />
              </svg>
            </a>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
