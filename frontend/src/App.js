import React, { useState, useEffect, useCallback } from 'react';

// Mock API functions (replace with your actual API calls)
const API_BASE_URL = 'http://localhost:8000';

const mockAnalyze = async (text) => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  // Simple mock sentiment analysis
  const positiveWords = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'happy', 'joy'];
  const negativeWords = ['bad', 'terrible', 'awful', 'hate', 'sad', 'angry', 'disappointed', 'frustrated'];
  
  const lowerText = text.toLowerCase();
  let score = 0.5; // neutral base
  
  positiveWords.forEach(word => {
    if (lowerText.includes(word)) score += 0.1;
  });
  
  negativeWords.forEach(word => {
    if (lowerText.includes(word)) score -= 0.1;
  });
  
  score = Math.max(0, Math.min(1, score)); // clamp between 0 and 1
  const label = score > 0.5 ? 'Positive' : 'Negative';
  const confidence = score > 0.7 || score < 0.3 ? 'High' : score > 0.6 || score < 0.4 ? 'Medium' : 'Low';
  
  return {
    label,
    score,
    confidence,
    cached: Math.random() > 0.7 // random cache hit
  };
};

const mockHealthCheck = async () => {
  return { status: 'healthy' };
};

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [health, setHealth] = useState(null);
  const [history, setHistory] = useState([]);
  const [toasts, setToasts] = useState([]);

  // Example texts for quick start
  const exampleTexts = [
    "I absolutely love this new product! It's amazing and works perfectly.",
    "This is terrible. I'm very disappointed with the service quality.",
    "The weather is okay today, nothing special but not bad either.",
    "What an incredible experience! I couldn't be happier with the results."
  ];

  // Load history from localStorage on mount
  useEffect(() => {
    const savedHistory = localStorage.getItem('sentiment-history');
    if (savedHistory) {
      try {
        setHistory(JSON.parse(savedHistory));
      } catch (e) {
        console.error('Failed to load history:', e);
      }
    }
  }, []);

  // Save history to localStorage whenever it changes
  useEffect(() => {
    if (history.length > 0) {
      localStorage.setItem('sentiment-history', JSON.stringify(history));
    }
  }, [history]);

  // Health check
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await mockHealthCheck();
        setHealth(response);
      } catch (err) {
        console.error('Health check failed:', err);
        setHealth({ status: 'unhealthy' });
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 60000);
    return () => clearInterval(interval);
  }, []);

  // Toast management
  const addToast = useCallback((message, type = 'info') => {
    const id = Date.now();
    const toast = { id, message, type };
    setToasts(prev => [...prev, toast]);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 3000);
  }, []);

  const removeToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  // Keyboard shortcut handler
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        if (!loading && text.trim()) {
          analyzeText();
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [text, loading]);

  const analyzeText = async () => {
    if (!text.trim()) {
      setError('Please enter text to analyze');
      addToast('Please enter text to analyze', 'error');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await mockAnalyze(text);
      setResult(response);
      
      // Add to history
      const historyItem = {
        id: Date.now(),
        text: text.trim(),
        result: response,
        timestamp: new Date().toISOString()
      };
      
      setHistory(prev => [historyItem, ...prev.slice(0, 9)]); // Keep last 10 items
      addToast('Analysis completed successfully!', 'success');
      
    } catch (err) {
      console.error('Analysis failed:', err);
      const errorMsg = err.response?.data?.detail || 'Analysis failed, please try again';
      setError(errorMsg);
      addToast(errorMsg, 'error');
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setText('');
    setResult(null);
    setError('');
    addToast('Content cleared', 'info');
  };

  const loadExampleText = (exampleText) => {
    setText(exampleText);
    addToast('Example text loaded', 'info');
  };

  const loadFromHistory = (historyItem) => {
    setText(historyItem.text);
    setResult(historyItem.result);
    addToast('Loaded from history', 'info');
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem('sentiment-history');
    addToast('History cleared', 'info');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Toast Container */}
      <div className="fixed top-4 right-4 z-50 space-y-2">
        {toasts.map(toast => (
          <div
            key={toast.id}
            className={`px-4 py-2 rounded-lg shadow-lg text-white transform transition-all duration-300 ease-in-out ${
              toast.type === 'success' ? 'bg-green-500' :
              toast.type === 'error' ? 'bg-red-500' :
              'bg-blue-500'
            }`}
            onClick={() => removeToast(toast.id)}
          >
            <div className="flex items-center space-x-2">
              <span>
                {toast.type === 'success' ? '✅' :
                 toast.type === 'error' ? '❌' : 'ℹ️'}
              </span>
              <span className="text-sm">{toast.message}</span>
              <button
                onClick={() => removeToast(toast.id)}
                className="ml-2 text-white hover:text-gray-200"
              >
                ×
              </button>
            </div>
          </div>
        ))}
      </div>

      <div className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            🤖 AI Sentiment Analysis Tool
          </h1>
          <p className="text-gray-600 mb-4">Enter text and AI will analyze its emotional sentiment</p>
          {health && (
            <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm ${
              health.status === 'healthy' 
                ? 'bg-green-100 text-green-800' 
                : 'bg-red-100 text-red-800'
            }`}>
              API Status: {health.status === 'healthy' ? '✅ Healthy' : '❌ Unhealthy'}
            </div>
          )}
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Analysis Section */}
          <div className="lg:col-span-2 space-y-6">
            {/* Example Texts */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-3">
                📝 Quick Start - Try Example Texts
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {exampleTexts.map((example, index) => (
                  <button
                    key={index}
                    onClick={() => loadExampleText(example)}
                    className="text-left p-3 text-sm bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors duration-200 border border-gray-200"
                    disabled={loading}
                  >
                    "{example.substring(0, 50)}..."
                  </button>
                ))}
              </div>
            </div>

            {/* Text Input Section */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Text to Analyze
                  </label>
                  <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Enter text to analyze sentiment..."
                    rows={6}
                    maxLength={500}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  />
                  <div className="flex justify-between items-center mt-2 text-sm">
                    <span className="text-gray-500">
                      {text.length}/500
                      {text.length >= 450 && (
                        <span className="text-orange-500 ml-1">- Approaching limit</span>
                      )}
                    </span>
                    <span className="text-gray-400">
                      Press Cmd/Ctrl + Enter to analyze
                    </span>
                  </div>
                </div>
                
                <div className="flex space-x-3">
                  <button 
                    onClick={analyzeText} 
                    disabled={loading || !text.trim()}
                    className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center"
                  >
                    {loading ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                        Analyzing...
                      </>
                    ) : 'Start Analysis'}
                  </button>
                  <button 
                    onClick={clearAll} 
                    className="bg-gray-500 hover:bg-gray-600 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200"
                    disabled={loading}
                  >
                    Clear
                  </button>
                </div>
              </div>

              {error && (
                <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                  <div className="flex items-center">
                    <span className="text-red-500 mr-2">❌</span>
                    <span className="text-red-700">{error}</span>
                  </div>
                </div>
              )}

              {result && (
                <div className="mt-6">
                  <h3 className="text-lg font-semibold text-gray-800 mb-3">📊 Analysis Results</h3>
                  <div className="bg-gray-50 rounded-lg p-4 space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="font-medium text-gray-700">Sentiment:</span>
                      <span className={`font-semibold ${
                        result.label === 'Positive' ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {result.label === 'Positive' ? '😊 Positive' : '😔 Negative'}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="font-medium text-gray-700">Confidence Score:</span>
                      <span className="font-semibold text-blue-600">
                        {(result.score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="font-medium text-gray-700">Reliability:</span>
                      <span className="font-semibold text-purple-600">{result.confidence}</span>
                    </div>
                    {result.cached && (
                      <div className="flex items-center justify-center mt-2 text-sm text-orange-600">
                        <span>⚡ Result from cache</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* History Sidebar */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-gray-800">
                📚 Analysis History
              </h3>
              {history.length > 0 && (
                <button
                  onClick={clearHistory}
                  className="text-sm text-red-600 hover:text-red-800"
                >
                  Clear All
                </button>
              )}
            </div>
            
            {history.length === 0 ? (
              <p className="text-gray-500 text-sm text-center py-8">
                No analysis history yet. Start by analyzing some text!
              </p>
            ) : (
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {history.map((item) => (
                  <div
                    key={item.id}
                    onClick={() => loadFromHistory(item)}
                    className="p-3 bg-gray-50 hover:bg-gray-100 rounded-lg cursor-pointer transition-colors duration-200 border border-gray-200"
                  >
                    <div className="text-sm">
                      <div className="font-medium text-gray-700 mb-1">
                        "{item.text.substring(0, 60)}..."
                      </div>
                      <div className="flex justify-between items-center text-xs text-gray-500">
                        <span className={`font-semibold ${
                          item.result.label === 'Positive' ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {item.result.label} ({(item.result.score * 100).toFixed(0)}%)
                        </span>
                        <span>
                          {new Date(item.timestamp).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center mt-12 text-gray-600">
          <p>Powered by FastAPI + React + Azure</p>
          <p className="text-sm">Version 1.0.0</p>
        </footer>
      </div>
    </div>
  );
}

export default App;