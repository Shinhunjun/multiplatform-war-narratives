import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || '';

const api = axios.create({ baseURL: API_BASE });

export type Platform = 'reddit' | 'news' | 'tiktok';

// Types
export interface OverviewStats {
  platform: string;
  total_documents: number;
  total_videos?: number;
  total_comments?: number;
  subreddits?: number;
  sources?: number;
  num_sources?: number;
  date_range: { start: string; end: string };
  num_topics: number;
  num_clusters: number;
  avg_sentiment: number;
  subreddit_list?: string[];
  source_list?: string[];
  all_months?: string[];
}

// TikTok-specific types
export interface HashtagTrend {
  hashtag: string;
  total_count: number;
  mean_sentiment: number;
}

export interface HashtagOverTime {
  year_month: string;
  hashtag: string;
  count: number;
  mean_sentiment: number;
}

export interface EngagementMetric {
  year_month: string;
  video_count: number;
  total_views: number;
  total_likes: number;
  total_shares: number;
  total_comments: number;
  avg_views: number;
  avg_likes: number;
  avg_duration: number;
}

export interface RegionDistribution {
  region_code: string;
  total_count: number;
}

export interface SentimentMonth {
  year_month: string;
  mean_sentiment: number;
  positive_ratio: number;
  negative_ratio: number;
  neutral_ratio: number;
  total_count: number;
}

export interface SentimentSubreddit {
  subreddit: string;
  mean_sentiment: number;
  positive_ratio: number;
  negative_ratio: number;
  neutral_ratio: number;
  total_count: number;
}

export interface SentimentSource {
  source: string;
  mean_sentiment: number;
  positive_ratio: number;
  negative_ratio: number;
  total_count: number;
}

export interface TopicInfo {
  Topic: number;
  Count: number;
  Name: string;
  Representation: string;
}

export interface TopicOverTime {
  Topic: number;
  Words: string;
  Frequency: number;
  Timestamp: string;
}

export interface TopicMonthly {
  year_month: string;
  topic_id: number;
  count: number;
  proportion: number;
  name: string;
  keywords: string;
}

export interface BoxPlotStat {
  subreddit: string;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  mean: number;
  std: number;
  count: number;
}

export interface ClusterMonthly {
  year_month: string;
  cluster_id: number;
  count: number;
  proportion: number;
  keywords: string;
}

export interface ClusterSummary {
  cluster_id: number;
  theme: string;
  summary: string;
  count: number;
  top_subreddit: string;
  sentiment_mean: number;
  time_start: string;
  time_end: string;
  keywords?: string;
  keywords_short?: string;
}

export interface ClusterScatterPoint {
  x: number;
  y: number;
  cluster_id: number;
  subreddit: string;
  keywords: string;
}

// API calls
export const fetchOverview = (platform?: Platform) => {
  const params = new URLSearchParams();
  if (platform) params.set('platform', platform);
  return api.get<OverviewStats>(`/api/overview/stats?${params}`).then(r => r.data);
};

export const fetchSentimentByMonth = (start?: string, end?: string, platform?: Platform) => {
  const params = new URLSearchParams();
  if (start) params.set('start', start);
  if (end) params.set('end', end);
  if (platform) params.set('platform', platform);
  return api.get<SentimentMonth[]>(`/api/sentiment/by-month?${params}`).then(r => r.data);
};

export const fetchSentimentBySubreddit = (platform?: Platform) => {
  const params = new URLSearchParams();
  if (platform) params.set('platform', platform);
  return api.get<SentimentSubreddit[]>(`/api/sentiment/by-subreddit?${params}`).then(r => r.data);
};

export const fetchSentimentBySubredditMonth = (subreddit?: string, start?: string, end?: string, platform?: Platform) => {
  const params = new URLSearchParams();
  if (subreddit) params.set('subreddit', subreddit);
  if (start) params.set('start', start);
  if (end) params.set('end', end);
  if (platform) params.set('platform', platform);
  return api.get<SentimentMonth[]>(`/api/sentiment/by-subreddit-month?${params}`).then(r => r.data);
};

export const fetchSentimentBoxplot = (start?: string, end?: string, platform?: Platform) => {
  const params = new URLSearchParams();
  if (start) params.set('start', start);
  if (end) params.set('end', end);
  if (platform) params.set('platform', platform);
  return api.get<BoxPlotStat[]>(`/api/sentiment/boxplot?${params}`).then(r => r.data);
};

export const fetchTopicInfo = (platform?: Platform, start?: string, end?: string) => {
  const params = new URLSearchParams();
  if (platform) params.set('platform', platform);
  if (start) params.set('start', start);
  if (end) params.set('end', end);
  return api.get<TopicInfo[]>(`/api/topics/info?${params}`).then(r => r.data);
};

export const fetchTopicsOverTime = (topicId?: number, platform?: Platform) => {
  const params = new URLSearchParams();
  if (topicId !== undefined) params.set('topic_id', String(topicId));
  if (platform) params.set('platform', platform);
  return api.get<TopicOverTime[]>(`/api/topics/over-time?${params}`).then(r => r.data);
};

export const fetchTopicsMonthly = (month: string, topN = 15, platform?: Platform) => {
  const params = new URLSearchParams();
  params.set('month', month);
  params.set('top_n', String(topN));
  if (platform) params.set('platform', platform);
  return api.get<TopicMonthly[]>(`/api/topics/monthly?${params}`).then(r => r.data);
};

export const fetchTopicsMonthlyMonths = (platform?: Platform) => {
  const params = new URLSearchParams();
  if (platform) params.set('platform', platform);
  return api.get<string[]>(`/api/topics/monthly/months?${params}`).then(r => r.data);
};

export interface TopicMonthlyFitted {
  year_month: string;
  topic_id: number;
  keywords: string;
  count: number;
  proportion: number;
}

export const fetchTopicsMonthlyFitted = (month: string, topN = 15, platform?: Platform) => {
  const params = new URLSearchParams();
  params.set('month', month);
  params.set('top_n', String(topN));
  if (platform) params.set('platform', platform);
  return api.get<TopicMonthlyFitted[]>(`/api/topics/monthly-fitted?${params}`).then(r => r.data);
};

export const fetchTopicsMonthlyFittedMonths = (platform?: Platform) => {
  const params = new URLSearchParams();
  if (platform) params.set('platform', platform);
  return api.get<string[]>(`/api/topics/monthly-fitted/months?${params}`).then(r => r.data);
};

export const fetchTopicsBySubreddit = (platform?: Platform) => {
  const params = new URLSearchParams();
  if (platform) params.set('platform', platform);
  return api.get<{ subreddit: string; topic_id: number; count: number; proportion: number }[]>(`/api/topics/by-subreddit?${params}`).then(r => r.data);
};

export const fetchClusterSummaries = (limit = 30, minCount = 20, start?: string, end?: string) => {
  const params = new URLSearchParams();
  params.set('limit', String(limit));
  params.set('min_count', String(minCount));
  if (start) params.set('start', start);
  if (end) params.set('end', end);
  return api.get<ClusterSummary[]>(`/api/clusters/summaries?${params}`).then(r => r.data);
};

export const fetchClusterScatter = (topN = 50, maxPoints = 30000, start?: string, end?: string) => {
  const params = new URLSearchParams();
  params.set('top_n', String(topN));
  params.set('max_points', String(maxPoints));
  if (start) params.set('start', start);
  if (end) params.set('end', end);
  return api.get<ClusterScatterPoint[]>(`/api/clusters/scatter?${params}`).then(r => r.data);
};

export const fetchClustersMonthly = (month: string, topN = 15) => {
  const params = new URLSearchParams();
  params.set('month', month);
  params.set('top_n', String(topN));
  return api.get<ClusterMonthly[]>(`/api/clusters/monthly?${params}`).then(r => r.data);
};

export const fetchClustersMonthlyMonths = () => {
  return api.get<string[]>(`/api/clusters/monthly/months`).then(r => r.data);
};

// TikTok-specific endpoints
export const fetchTikTokHashtags = (topN = 20, start?: string, end?: string) => {
  const params = new URLSearchParams();
  params.set('top_n', String(topN));
  if (start) params.set('start', start);
  if (end) params.set('end', end);
  return api.get<HashtagTrend[]>(`/api/tiktok/hashtags?${params}`).then(r => r.data);
};

export const fetchTikTokHashtagsOverTime = (hashtags?: string, topN = 10) => {
  const params = new URLSearchParams();
  params.set('top_n', String(topN));
  if (hashtags) params.set('hashtags', hashtags);
  return api.get<HashtagOverTime[]>(`/api/tiktok/hashtags/over-time?${params}`).then(r => r.data);
};

export const fetchTikTokEngagement = (start?: string, end?: string) => {
  const params = new URLSearchParams();
  if (start) params.set('start', start);
  if (end) params.set('end', end);
  return api.get<EngagementMetric[]>(`/api/tiktok/engagement?${params}`).then(r => r.data);
};

export const fetchTikTokRegions = (topN = 15, start?: string, end?: string) => {
  const params = new URLSearchParams();
  params.set('top_n', String(topN));
  if (start) params.set('start', start);
  if (end) params.set('end', end);
  return api.get<RegionDistribution[]>(`/api/tiktok/regions?${params}`).then(r => r.data);
};

// Reports
export interface Report {
  period: string;
  generated_at?: string;
  report?: string;
  error?: string;
  data_summary?: Record<string, any>;
}

export const fetchReport = (start: string, end: string, force = false) => {
  const params = new URLSearchParams();
  params.set('start', start);
  params.set('end', end);
  if (force) params.set('force', 'true');
  return api.get<Report>(`/api/reports/generate?${params}`).then(r => r.data);
};

export const fetchReportList = () =>
  api.get<{ period: string; generated_at: string; has_error: boolean }[]>('/api/reports/list').then(r => r.data);

// Chat
export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export const sendChatMessage = (question: string, history?: ChatMessage[]) =>
  api.post<{ answer: string }>('/api/chat', { question, history }).then(r => r.data);

export const fetchTemporalClusters = (limit = 10, start?: string, end?: string) => {
  const params = new URLSearchParams();
  params.set('limit', String(limit));
  if (start) params.set('start', start);
  if (end) params.set('end', end);
  return api.get<{ year_month: string; cluster_id: number; count: number; proportion: number }[]>(`/api/clusters/temporal?${params}`).then(r => r.data);
};
