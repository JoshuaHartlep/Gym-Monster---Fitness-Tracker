-- SQL script to set up the weight_logs table in Supabase
-- Run this in your Supabase SQL editor

-- Create the weight_logs table
CREATE TABLE IF NOT EXISTS weight_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    date DATE NOT NULL,
    weight DECIMAL(5,2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure unique entries per user per date
    UNIQUE(user_id, date)
);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_weight_logs_user_date ON weight_logs(user_id, date);

-- Enable Row Level Security (RLS)
ALTER TABLE weight_logs ENABLE ROW LEVEL SECURITY;

-- Create policy to allow users to only see their own data
CREATE POLICY "Users can view own weight logs" ON weight_logs
    FOR SELECT USING (auth.uid() = user_id);

-- Create policy to allow users to insert their own data
CREATE POLICY "Users can insert own weight logs" ON weight_logs
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Create policy to allow users to update their own data
CREATE POLICY "Users can update own weight logs" ON weight_logs
    FOR UPDATE USING (auth.uid() = user_id);

-- Create policy to allow users to delete their own data
CREATE POLICY "Users can delete own weight logs" ON weight_logs
    FOR DELETE USING (auth.uid() = user_id);

-- Create function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_weight_logs_updated_at 
    BEFORE UPDATE ON weight_logs 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();
