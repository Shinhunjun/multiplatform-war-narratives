import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Sidebar from './components/layout/Sidebar';
import Dashboard from './pages/Dashboard';
import SentimentPage from './pages/SentimentPage';
import TopicsPage from './pages/TopicsPage';
import ClustersPage from './pages/ClustersPage';

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen">
        <Sidebar />
        <main className="flex-1 overflow-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/sentiment" element={<SentimentPage />} />
            <Route path="/topics" element={<TopicsPage />} />
            <Route path="/clusters" element={<ClustersPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
