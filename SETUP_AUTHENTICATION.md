# Gym Monster Authentication Setup

This document explains how to set up Supabase authentication for the Gym Monster app.

## Prerequisites

1. A Supabase account and project
2. The app dependencies installed (`pip install -r requirements.txt`)

## Setup Steps

### 1. Create Supabase Project

1. Go to [supabase.com](https://supabase.com) and create a new project
2. Note down your project URL and anon key

### 2. Set Up Database Table

1. In your Supabase dashboard, go to the SQL Editor
2. Run the SQL script from `setup_supabase.sql` to create the required table and policies

### 3. Configure Authentication

1. In Supabase dashboard, go to Authentication → Settings
2. Enable email authentication
3. **Enable Google OAuth**:
   - Go to Authentication → Providers
   - Enable Google provider
   - Add your Google OAuth credentials (Client ID and Client Secret)
   - Set the redirect URL to your deployed app URL (e.g., `https://gym-monster.streamlit.app`)
4. Configure any additional settings as needed (password requirements, etc.)

### 4. Set Up Streamlit Secrets

#### For Local Development:
1. Create a `.streamlit/secrets.toml` file (copy from `.streamlit/secrets.toml.example`)
2. Add your Supabase credentials:
```toml
SUPABASE_URL = "your-project-url"
SUPABASE_ANON_KEY = "your-anon-key"
```

#### For Streamlit Cloud:
1. Go to your app settings in Streamlit Cloud
2. Add the secrets in the "Secrets" section:
```toml
SUPABASE_URL = "your-project-url"
SUPABASE_ANON_KEY = "your-anon-key"
```

## Features

### Authentication Modes

1. **Google OAuth**: Quick login with Google account (recommended)
2. **Sign Up**: Users can create new accounts with email/password
3. **Login**: Existing users can sign in with their credentials
4. **Guest Mode**: Users can explore the app without creating an account (data not persisted)

### Data Persistence

- **Authenticated Users**: Data is stored in Supabase and persists across sessions
- **Guest Users**: Data is stored locally in JSON file (resets on refresh)

### Security

- Row Level Security (RLS) ensures users can only access their own data
- Passwords are handled securely by Supabase Auth
- Guest mode provides a safe demo experience

## Google OAuth Setup

### Prerequisites for Google OAuth

1. **Google Cloud Console Setup**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select existing one
   - Enable the Google+ API
   - Go to "Credentials" → "Create Credentials" → "OAuth 2.0 Client IDs"
   - Set application type to "Web application"
   - Add authorized redirect URIs:
     - For local development: `http://localhost:8501`
     - For production: `https://your-app-url.streamlit.app`

2. **Supabase Configuration**:
   - In Supabase dashboard, go to Authentication → Providers
   - Enable Google provider
   - Enter your Google OAuth Client ID and Client Secret
   - Set the redirect URL to your deployed app URL

### OAuth Flow Types

The app supports both OAuth flow types:

1. **PKCE Flow (Recommended - More Secure)**:
   - Authorization code is returned in query params: `?code=...`
   - Code is exchanged server-side for tokens
   - Tokens never appear in browser URL
   - To enable: Set `flow_type=pkce` in OAuth URL (already configured)
   - **Note**: Supabase may default to implicit flow - check dashboard settings

2. **Implicit Flow (Fallback - Less Secure)**:
   - Tokens returned in URL fragment: `#access_token=...`
   - JavaScript extracts tokens and passes to Streamlit
   - Tokens briefly visible in browser URL
   - Automatically detected if Supabase uses this flow

**Current Implementation**: The app automatically handles both flows. If PKCE doesn't work (Supabase returns tokens in fragment), the app falls back to implicit flow with JavaScript extraction.

### Testing Google OAuth

1. Run the app: `streamlit run app.py`
2. Click the "Login with Google" button in the sidebar
3. Complete the Google OAuth flow
4. Verify you're logged in and data persists

## Testing

1. Run the app: `streamlit run app.py`
2. Test each authentication mode:
   - **Google OAuth**: Click "Login with Google" in sidebar
   - Create a new account with email/password
   - Log in with existing credentials
   - Try guest mode
3. Verify data persistence works correctly for each mode

## Troubleshooting

### Common Issues

1. **Supabase connection failed**: Check your URL and anon key in secrets
2. **Table not found**: Run the SQL setup script in Supabase
3. **Authentication errors**: Check Supabase Auth settings
4. **RLS policy errors**: Verify the policies are correctly set up
5. **Google OAuth not working**:
   - Verify Google OAuth credentials are correctly set in Supabase
   - Check that redirect URLs match exactly (including protocol and port)
   - Ensure Google+ API is enabled in Google Cloud Console
   - Verify the OAuth consent screen is configured

### Debug Mode

If you encounter issues, the app will show a warning and fall back to guest mode only.

## File Structure

- `app.py`: Main application with authentication integration
- `setup_supabase.sql`: Database setup script
- `.streamlit/secrets.toml.example`: Example secrets configuration
- `requirements.txt`: Python dependencies (includes supabase package)
