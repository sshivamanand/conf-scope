import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Target, Upload, FileText, Loader2, TrendingUp } from 'lucide-react';

function Predict() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
      setError('');
    } else {
      setError('Please upload a valid PDF file');
      setFile(null);
    }
  };

  const handleSubmit = async () => {
    if (!file) {
      setError('Please upload a PDF file');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
      });
      
      const data = await res.json();
      
      if (res.ok) {
        // Navigate to /result route on successful upload
        navigate('/result', { state: { filename: data.filename } });
      } else {
        setError(data.error || 'Failed to upload file');
      }
    } catch (err) {
      setError('Failed to get predictions. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setFile(null);
    setError('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <nav className="container mx-auto px-6 py-6">
        <div className="flex items-center justify-between">
          <Link to="/" className="flex items-center space-x-2 cursor-pointer">
            <Target className="w-8 h-8 text-purple-400" />
            <span className="text-2xl font-bold text-white hover:text-purple-300 transition-colors">
              ConfScope
            </span>
          </Link>
        </div>
      </nav>

      <div className="container mx-auto px-6 py-12">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4 text-center">
            Predict Acceptance Probability
          </h1>
          <p className="text-gray-400 text-center mb-12 max-w-2xl mx-auto">
            Upload your completed conference paper (PDF format). The system will automatically extract the abstract and predict acceptance probabilities across major conferences.
          </p>

          <div className="bg-white/5 backdrop-blur-lg rounded-xl p-8 border border-white/10 mb-8">
            <div className="mb-6">
              <label className="block text-white font-semibold mb-3">
                Upload Conference Paper (PDF)
              </label>
              <div className="relative">
                <input
                  type="file"
                  accept=".pdf"
                  onChange={handleFileChange}
                  className="hidden"
                  id="pdf-upload"
                />
                <label
                  htmlFor="pdf-upload"
                  className="flex flex-col items-center justify-center space-y-3 bg-white/10 border-2 border-dashed border-purple-400/50 rounded-lg p-12 cursor-pointer hover:bg-white/15 transition-all duration-300"
                >
                  <Upload className="w-12 h-12 text-purple-400" />
                  <span className="text-gray-300 text-lg">
                    {file ? file.name : 'Click to upload your paper'}
                  </span>
                  <span className="text-gray-500 text-sm">
                    PDF format • Abstract will be extracted automatically
                  </span>
                </label>
              </div>
              {file && (
                <div className="mt-3 flex items-center space-x-2 text-green-400">
                  <FileText className="w-5 h-5" />
                  <span>Ready to analyze: {file.name}</span>
                </div>
              )}
            </div>

            {error && (
              <div className="mb-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300">
                {error}
              </div>
            )}

            <div className="flex space-x-4">
              <button
                onClick={handleSubmit}
                disabled={loading}
                className="flex-1 flex items-center justify-center space-x-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white px-6 py-4 rounded-lg font-semibold hover:from-purple-600 hover:to-pink-600 transition-all duration-300 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <>
                    <TrendingUp className="w-5 h-5" />
                    <span>Get Predictions</span>
                  </>
                )}
              </button>
              <button
                onClick={handleClear}
                className="px-6 py-4 bg-white/10 text-white rounded-lg font-semibold hover:bg-white/20 transition-all duration-300"
              >
                Clear
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Predict;