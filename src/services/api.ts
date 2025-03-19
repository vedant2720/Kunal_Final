
import { DeepfakeResult } from "@/lib/types";

// Point to your Flask API endpoint
const API_URL = "http://localhost:5000";

export async function detectDeepfake(file: File): Promise<DeepfakeResult> {
  try {
    const formData = new FormData();
    
    // Determine if the file is an image or video based on its type
    const isVideo = file.type.startsWith('video/');
    
    // Add the file to the form data with the appropriate key
    formData.append(isVideo ? "video" : "image", file);

    // Use the appropriate endpoint based on the file type
    const endpoint = isVideo ? "detect_video" : "detect";
    
    const response = await fetch(`${API_URL}/api/${endpoint}`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `Failed to analyze ${isVideo ? "video" : "image"}`);
    }

    return await response.json();
  } catch (error) {
    console.error("API Error:", error);
    throw error;
  }
}

// This function can be used to check if the API server is running
export async function checkApiStatus(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/api/status`, {
      method: "GET",
    });
    return response.ok;
  } catch (error) {
    console.error("API Status Check Error:", error);
    return false;
  }
}
