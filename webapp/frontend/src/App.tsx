import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { TimeRangeProvider } from './lib/TimeRangeContext';
import TimeRangeBar from './components/TimeRangeBar';
import Sidebar from './components/layout/Sidebar';
import Dashboard from './pages/Dashboard';
import SentimentPage from './pages/SentimentPage';
import TopicsPage from './pages/TopicsPage';
import ClustersPage from './pages/ClustersPage';
import TikTokPage from './pages/TikTokPage';
import ReportsPage from './pages/ReportsPage';
import ChatPage from './pages/ChatPage';

export default function App() {
  return (
    <BrowserRouter>
      <TimeRangeProvider>
        <div className="flex min-h-screen bg-[#0f1117]">
          <Sidebar />
          <main className="flex-1 min-w-0">
            <TimeRangeBar />
            <div className="max-w-[1400px] mx-auto">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/sentiment" element={<SentimentPage />} />
                <Route path="/topics" element={<TopicsPage />} />
                <Route path="/tiktok" element={<TikTokPage />} />
                <Route path="/clusters" element={<ClustersPage />} />
                <Route path="/reports" element={<ReportsPage />} />
                <Route path="/chat" element={<ChatPage />} />
              </Routes>
            </div>
          </main>
        </div>
      </TimeRangeProvider>
    </BrowserRouter>
  );
}
