
import { DeepfakeResult } from "@/lib/types";

// This would be your actual Flask API endpoint
const API_URL = "http://localhost:5000";

export async function detectDeepfake(image: File): Promise<DeepfakeResult> {
  try {
    const formData = new FormData();
    formData.append("image", image);

    const response = await fetch(`${API_URL}/api/detect`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || "Failed to analyze image");
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
