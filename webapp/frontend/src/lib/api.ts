import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || '';

const api = axios.create({ baseURL: API_BASE });

export type Platform = 'reddit' | 'news';

// Types
export interface OverviewStats {
  platform: string;
  total_documents: number;
  subreddits?: number;
  sources?: number;
  date_range: { start: string; end: string };
  num_topics: number;
  num_clusters: number;
  avg_sentiment: number;
  subreddit_list?: string[];
  source_list?: string[];
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

export const fetchTemporalClusters = (limit = 10, start?: string, end?: string) => {
  const params = new URLSearchParams();
  params.set('limit', String(limit));
  if (start) params.set('start', start);
  if (end) params.set('end', end);
  return api.get<{ year_month: string; cluster_id: number; count: number; proportion: number }[]>(`/api/clusters/temporal?${params}`).then(r => r.data);
};
