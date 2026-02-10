import React, { useState, useEffect } from 'react';
import { Target, TrendingUp, Award } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { useLocation, useNavigate } from 'react-router-dom';

function Results() {
  const location = useLocation();
  const navigate = useNavigate();

  // Use passed state immediately if available
  const [results, setResults] = useState(location.state?.results || null);
  const [loading, setLoading] = useState(!location.state?.results);

  // Fallback fetch only if no state passed (page refresh case)
  useEffect(() => {
    if (!location.state?.results) {
      fetch('http://127.0.0.1:5000/result')
        .then(res => res.json())
        .then(data => {
          if (data && data.prediction) {
            setResults(data);
          }
          setLoading(false);
        })
        .catch(err => {
          console.error('Error fetching results:', err);
          setLoading(false);
        });
    }
  }, [location.state]);

  // Loading screen
  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="text-white text-2xl">Loading results...</div>
      </div>
    );
  }

  // No results screen
  if (!results || !results.prediction) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <p className="text-white text-xl mb-4">No prediction available</p>
          <button
            onClick={() => navigate('/predict')}
            className="text-purple-400 hover:text-purple-300"
          >
            Go back to upload
          </button>
        </div>
      </div>
    );
  }

  // Map binary labels to conference names
  const conferenceMap = {
    '001': 'ICLR',
    '010': 'CoNLL',
    '100': 'ACL',
    '000': 'Other'
  };

  // Format predictions
  const conferences = Object.entries(results.prediction)
    .map(([code, probability]) => ({
      name: conferenceMap[code] || code,
      probability: probability * 100,
      value: probability,
      code: code
    }))
    .filter(conf => conf.name !== 'Other')
    .sort((a, b) => b.probability - a.probability);

  const colors = ['#a78bfa', '#ec4899', '#8b5cf6'];

  const pieData = conferences.map((conf, idx) => ({
    name: conf.name,
    value: conf.probability,
    color: colors[idx % colors.length]
  }));

  const CircularProgress = ({ percentage, conference, color }) => {
    const radius = 60;
    const circumference = 2 * Math.PI * radius;
    const strokeDashoffset = circumference - (percentage / 100) * circumference;

    return (
      <div className="bg-white/5 backdrop-blur-lg rounded-xl p-6 border border-white/10 hover:border-purple-400/50 transition-all duration-300">
        <div className="flex flex-col items-center">
          <div className="relative w-40 h-40 mb-4">
            <svg className="transform -rotate-90 w-40 h-40">
              <circle
                cx="80"
                cy="80"
                r={radius}
                stroke="rgba(255,255,255,0.1)"
                strokeWidth="12"
                fill="none"
              />
              <circle
                cx="80"
                cy="80"
                r={radius}
                stroke={color}
                strokeWidth="12"
                fill="none"
                strokeDasharray={circumference}
                strokeDashoffset={strokeDashoffset}
                strokeLinecap="round"
                className="transition-all duration-1000 ease-out"
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-3xl font-bold text-white">
                {percentage.toFixed(1)}%
              </span>
            </div>
          </div>
          <h3 className="text-xl font-semibold text-white">{conference}</h3>
          <p className="text-gray-400 text-sm">Acceptance Probability</p>
        </div>
      </div>
    );
  };

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-800 border border-purple-400/50 rounded-lg p-3 shadow-lg">
          <p className="text-white font-semibold">{payload[0].name}</p>
          <p className="text-purple-300">{payload[0].value.toFixed(1)}%</p>
        </div>
      );
    }
    return null;
  };

  const topConference = conferences[0];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">

      {/* Navigation */}
      <nav className="container mx-auto px-6 py-6">
        <div className="flex items-center space-x-2">
          <Target className="w-8 h-8 text-purple-400" />
          <span className="text-2xl font-bold text-white">ConfScope</span>
        </div>
      </nav>

      {/* Top Recommendation */}
      <div className="container mx-auto px-6 py-12 max-w-4xl">
        <div className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 backdrop-blur-lg rounded-xl p-8 border border-purple-400/50 mb-12">
          <div className="flex items-center space-x-3 mb-4">
            <Award className="w-8 h-8 text-yellow-400" />
            <h3 className="text-2xl font-bold text-white">Top Recommendation</h3>
          </div>

          <div className="flex justify-between items-center">
            <div>
              <p className="text-4xl font-bold text-white">{topConference.name}</p>
              <p className="text-gray-300">Highest predicted acceptance rate</p>
            </div>

            <p className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
              {topConference.probability.toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Pie Chart */}
        <div className="bg-white/5 backdrop-blur-lg rounded-xl p-8 border border-white/10 mb-12">
          <div className="flex items-center space-x-3 mb-6">
            <TrendingUp className="w-6 h-6 text-purple-400" />
            <h3 className="text-2xl font-bold text-white">Overall Distribution</h3>
          </div>

          <ResponsiveContainer width="100%" height={400}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" outerRadius={120} dataKey="value">
                {pieData.map((entry, index) => (
                  <Cell key={index} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Circular Progress */}
        <div className="grid md:grid-cols-3 gap-6">
          {conferences.map((conf, idx) => (
            <CircularProgress
              key={conf.name}
              percentage={conf.probability}
              conference={conf.name}
              color={colors[idx % colors.length]}
            />
          ))}
        </div>

      </div>

    </div>
  );
}

export default Results;
