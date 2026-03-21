import { useState } from "react";
import "./GarbageDetectionPage.css";

interface Detection {
  class: string;
  confidence: number;
}

interface DetectionResult {
  num_detections: number;
  detections: Detection[];
  annotated_image: string;
}

const GarbageDetectionPage = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<DetectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedImage) {
      setError("Please select an image first");
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append("file", selectedImage);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to process image");
      }

      const data: DetectionResult = await response.json();
      setResults(data);
    } catch (err: any) {
      setError(err.message || "Error processing image");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    setResults(null);
    setError(null);
  };

  return (
    <div className="gd-container">
      <div className="gd-header">
        <h2>🗑️ Garbage Object Detection</h2>
        <p>Upload an image to detect and classify garbage objects</p>
      </div>

      <div className="gd-upload-section">
        <div className="gd-file-input-wrapper">
          <input
            type="file"
            id="gd-file-input"
            accept="image/*"
            onChange={handleImageSelect}
            className="gd-file-input"
          />
          <label htmlFor="gd-file-input" className="gd-file-label">
            {selectedImage ? "📁 Change Image" : "📁 Choose Image"}
          </label>
          {selectedImage && (
            <span className="gd-file-name">{selectedImage.name}</span>
          )}
        </div>

        <div className="gd-action-buttons">
          <button
            onClick={handleUpload}
            disabled={!selectedImage || loading}
            className="gd-btn gd-btn-primary"
          >
            {loading ? "🔄 Processing..." : "🚀 Detect Objects"}
          </button>
          <button
            onClick={handleReset}
            disabled={loading}
            className="gd-btn gd-btn-secondary"
          >
            🔄 Reset
          </button>
        </div>
      </div>

      {error && <div className="gd-error-message">⚠️ {error}</div>}

      <div className="gd-results-section">
        {previewUrl && !results && (
          <div className="gd-image-preview">
            <h3>Original Image</h3>
            <img src={previewUrl} alt="Preview" />
          </div>
        )}

        {results && (
          <>
            <div className="gd-detections-info">
              <h3>Detection Results</h3>
              <p className="gd-detection-count">
                Found <strong>{results.num_detections}</strong> object(s)
              </p>

              {results.detections.length > 0 && (
                <div className="gd-detections-list">
                  {results.detections.map((detection, index) => (
                    <div key={index} className="gd-detection-item">
                      <span className="gd-detection-class">
                        {detection.class}
                      </span>
                      <span className="gd-detection-confidence">
                        {(detection.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="gd-image-preview">
              <h3>Detected Objects</h3>
              <img src={results.annotated_image} alt="Detected objects" />
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default GarbageDetectionPage;
