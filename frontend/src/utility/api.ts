const base_url = "http://localhost:3000";
import {
  RegisterDetails,
  RegisterResponse,
  LoginDetails,
  LoginResponse,
  GarbageReportsResponse,
} from "../Model/model";

export const predictWaste = async (file: File) => {
  console.log("file", file);
  try {
    if (!file) {
      throw new Error("Please upload an image first!");
    }
    const formdata = new FormData();
    formdata.append("file", file);

    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formdata,
    });
    if (!response.ok) {
      throw new Error("Failed to get prediction");
    }
    const data = await response.json();
    console.log(".....>", data);
    return { success: true, data: data, message: "Predict succesfully" };
  } catch (error: any) {
    console.error("Register error:", error);
    return { success: false, message: error.message || "Prediction failed" };
  }
};

export const askQuestion = async (question: string): Promise<string> => {
  try {
    const response = await fetch(`${base_url}/api/ask/question`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) {
      throw new Error("Failed to fetch response");
    }
    const data = await response.json();
    return data.answer || "No answer found.";
  } catch (error) {
    console.error("Error fetching AI response:", error);
    return "⚠️ Error fetching answer. Please try again.";
  }
};

export const userRegister = async (
  data: RegisterDetails
): Promise<RegisterResponse> => {
  try {
    const res = await fetch(`${base_url}/api/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    const result = await res.json();

    if (!res.ok || !result.success) {
      throw new Error(result.message || "Registration failed!");
    }

    // Store user details and token in localStorage
    if (result.data) {
      localStorage.setItem("token", result.data.token);
      localStorage.setItem("user", JSON.stringify(result.data.user));
    }

    return {
      success: true,
      message: result.message || "Registration successful!",
      data: result.data || null,
    };
  } catch (error: any) {
    console.error("Register error:", error);
    return { success: false, message: error.message || "Registration failed" };
  }
};

export const userLogin = async (data: LoginDetails): Promise<LoginResponse> => {
  try {
    const res = await fetch(`${base_url}/api/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    const result = await res.json();

    if (!res.ok || !result.success) {
      throw new Error(result.message || "Login failed!");
    }

    // Store user details and token in localStorage
    if (result.data) {
      localStorage.setItem("token", result.data.token);
      localStorage.setItem("user", JSON.stringify(result.data.user));
    }

    return {
      success: true,
      message: result.message || "Login successful!",
      data: result.data,
      token: result.data?.token,
    };
  } catch (error: any) {
    console.error("Login error:", error);
    return { success: false, message: error.message || "Login failed" };
  }
};

export const getCollectorGarbageReports = async (
  collectorId: string
): Promise<GarbageReportsResponse> => {
  try {
    const response = await fetch(
      `${base_url}/api/garbage/reports/collector/${collectorId}`
    );
    if (!response.ok) {
      throw new Error("Failed to fetch garbage reports");
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error fetching collector garbage reports:", error);
    throw error;
  }
};

export interface CreateGarbageReportData {
  reporterName: string;
  weight: number;
  collectionDeadline: string;
  additionalDetails: string;
  latitude: number;
  longitude: number;
  address: string;
  garbageImage?: File;
}

export const createGarbageReport = async (data: CreateGarbageReportData) => {
  try {
    const formData = new FormData();
    formData.append("reporterName", data.reporterName);
    formData.append("weight", data.weight.toString());
    formData.append("collectionDeadline", data.collectionDeadline);
    formData.append("additionalDetails", data.additionalDetails);
    formData.append("latitude", data.latitude.toString());
    formData.append("longitude", data.longitude.toString());
    formData.append("address", data.address);
    if (data.garbageImage) {
      formData.append("garbageImage", data.garbageImage);
    }

    const response = await fetch(`${base_url}/api/garbage/reports`, {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (!response.ok || !result.success) {
      throw new Error(result.message || "Failed to submit garbage report");
    }

    return { success: true, message: result.message, data: result.data };
  } catch (error: any) {
    console.error("Error creating garbage report:", error);
    return {
      success: false,
      message: error.message || "Failed to submit garbage report",
    };
  }
};

// Utility functions for localStorage
export const getCurrentUser = () => {
  const user = localStorage.getItem("user");
  return user ? JSON.parse(user) : null;
};

export const getStoredToken = () => {
  return localStorage.getItem("token");
};

export const logout = () => {
  localStorage.removeItem("user");
  localStorage.removeItem("token");
};

// Chat APIs
export const getChatUsers = async (currentUserId: string) => {
  try {
    const response = await fetch(`${base_url}/api/chat/contacts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ currentUserId }),
    });
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error fetching chat contacts:", error);
    return { success: false, message: "Error fetching contacts" };
  }
};

export const getAllUsers = async (currentUserId: string) => {
  try {
    const response = await fetch(`${base_url}/api/chat/users`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ currentUserId }),
    });
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error fetching all users:", error);
    return { success: false, message: "Error fetching users" };
  }
};

export const getChatHistory = async (userId: string, currentUserId: string) => {
  try {
    const response = await fetch(`${base_url}/api/chat/history/${userId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ currentUserId }),
    });
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error fetching chat history:", error);
    return { success: false, message: "Error fetching history" };
  }
};

export const sendMessage = async (
  sender: string,
  receiver: string,
  message: string
) => {
  try {
    const response = await fetch(`${base_url}/api/chat/send`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sender, receiver, message }),
    });
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error sending message:", error);
    return { success: false, message: "Error sending message" };
  }
};

export const getReportForCollector = async (collector_id: string) => {
  try {
   
    if (!collector_id) {
      console.log("Please provide the collector ID");
      return { success: false, message: "Collector ID is required" };
    }

    const response = await fetch(`http://localhost:3000/api/garbage/reports-assigned/${collector_id}`,
      {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      }
    );

    const data = await response.json();
    return data; 

  } catch (err: any) {
    console.error("Error fetching reports:", err);
    return { success: false, message: `Error msg: ${err.message}` };
  }
};



export const getUsers = async (user_id: string) => {
  try {
   
    if (!user_id) {
      console.log("Please provide the collector ID");
      return { success: false, message: "Collector ID is required" };
    }

    const response = await fetch(`http://localhost:3000/api/auth/getUser/${user_id}`,
      {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      }
    );

    const data = await response.json();
    return data; 

  } catch (err: any) {
    console.error("Error fetching reports:", err);
    return { success: false, message: `Error msg: ${err.message}` };
  }
};

