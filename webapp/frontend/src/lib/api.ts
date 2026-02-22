import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || '';

const api = axios.create({ baseURL: API_BASE });

// Types
export interface OverviewStats {
  platform: string;
  total_documents: number;
  subreddits: number;
  date_range: { start: string; end: string };
  num_topics: number;
  num_clusters: number;
  avg_sentiment: number;
  subreddit_list: string[];
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

export interface ClusterSummary {
  cluster_id: number;
  theme: string;
  summary: string;
  count: number;
  top_subreddit: string;
  sentiment_mean: number;
  time_start: string;
  time_end: string;
}

// API calls
export const fetchOverview = () =>
  api.get<OverviewStats>('/api/overview/stats').then(r => r.data);

export const fetchSentimentByMonth = (start?: string, end?: string) => {
  const params = new URLSearchParams();
  if (start) params.set('start', start);
  if (end) params.set('end', end);
  return api.get<SentimentMonth[]>(`/api/sentiment/by-month?${params}`).then(r => r.data);
};

export const fetchSentimentBySubreddit = () =>
  api.get<SentimentSubreddit[]>('/api/sentiment/by-subreddit').then(r => r.data);

export const fetchSentimentBySubredditMonth = (subreddit?: string, start?: string, end?: string) => {
  const params = new URLSearchParams();
  if (subreddit) params.set('subreddit', subreddit);
  if (start) params.set('start', start);
  if (end) params.set('end', end);
  return api.get<SentimentMonth[]>(`/api/sentiment/by-subreddit-month?${params}`).then(r => r.data);
};

export const fetchSentimentBoxplot = (start?: string, end?: string) => {
  const params = new URLSearchParams();
  if (start) params.set('start', start);
  if (end) params.set('end', end);
  return api.get<BoxPlotStat[]>(`/api/sentiment/boxplot?${params}`).then(r => r.data);
};

export const fetchTopicInfo = () =>
  api.get<TopicInfo[]>('/api/topics/info').then(r => r.data);

export const fetchTopicsOverTime = (topicId?: number) => {
  const params = topicId !== undefined ? `?topic_id=${topicId}` : '';
  return api.get<TopicOverTime[]>(`/api/topics/over-time${params}`).then(r => r.data);
};

export const fetchTopicsBySubreddit = () =>
  api.get<{ subreddit: string; topic_id: number; count: number; proportion: number }[]>('/api/topics/by-subreddit').then(r => r.data);

export const fetchClusterSummaries = (limit = 30, minCount = 20) =>
  api.get<ClusterSummary[]>(`/api/clusters/summaries?limit=${limit}&min_count=${minCount}`).then(r => r.data);

export const fetchTemporalClusters = (limit = 10, start?: string, end?: string) => {
  const params = new URLSearchParams();
  params.set('limit', String(limit));
  if (start) params.set('start', start);
  if (end) params.set('end', end);
  return api.get<{ year_month: string; cluster_id: number; count: number; proportion: number }[]>(`/api/clusters/temporal?${params}`).then(r => r.data);
};
