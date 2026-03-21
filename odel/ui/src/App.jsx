import { useState } from 'react'
import './App.css'

function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  const handleImageSelect = (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedImage(file)
      setPreviewUrl(URL.createObjectURL(file))
      setResults(null)
      setError(null)
    }
  }

  const handleUpload = async () => {
    if (!selectedImage) {
      setError('Please select an image first')
      return
    }

    setLoading(true)
    setError(null)
    setResults(null)

    const formData = new FormData()
    formData.append('file', selectedImage)

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Failed to process image')
      }

      const data = await response.json()
      setResults(data)
    } catch (err) {
      setError(err.message || 'Error processing image')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setSelectedImage(null)
    setPreviewUrl(null)
    setResults(null)
    setError(null)
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🗑️ Object Detection</h1>
        <p>Upload an image to detect objects</p>
      </header>

      <main className="main-content">
        <div className="upload-section">
          <div className="file-input-wrapper">
            <input
              type="file"
              id="file-input"
              accept="image/*"
              onChange={handleImageSelect}
              className="file-input"
            />
            <label htmlFor="file-input" className="file-label">
              {selectedImage ? '📁 Change Image' : '📁 Choose Image'}
            </label>
          </div>

          <div className="action-buttons">
            <button
              onClick={handleUpload}
              disabled={!selectedImage || loading}
              className="btn btn-primary"
            >
              {loading ? '🔄 Processing...' : '🚀 Detect Objects'}
            </button>
            <button
              onClick={handleReset}
              disabled={loading}
              className="btn btn-secondary"
            >
              🔄 Reset
            </button>
          </div>
        </div>

        {error && (
          <div className="error-message">
            ⚠️ {error}
          </div>
        )}

        <div className="results-section">
          {previewUrl && !results && (
            <div className="image-preview">
              <h3>Original Image</h3>
              <img src={previewUrl} alt="Preview" />
            </div>
          )}

          {results && (
            <>
              <div className="detections-info">
                <h3>Detection Results</h3>
                <p className="detection-count">
                  Found <strong>{results.num_detections}</strong> object(s)
                </p>
                
                {results.detections.length > 0 && (
                  <div className="detections-list">
                    {results.detections.map((detection, index) => (
                      <div key={index} className="detection-item">
                        <span className="detection-class">{detection.class}</span>
                        <span className="detection-confidence">
                          {(detection.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="image-preview">
                <h3>Detected Objects</h3>
                <img src={results.annotated_image} alt="Detected objects" />
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
