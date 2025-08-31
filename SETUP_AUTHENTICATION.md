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
3. Configure any additional settings as needed (password requirements, etc.)

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

1. **Sign Up**: Users can create new accounts with email/password
2. **Login**: Existing users can sign in with their credentials
3. **Guest Mode**: Users can explore the app without creating an account (data not persisted)

### Data Persistence

- **Authenticated Users**: Data is stored in Supabase and persists across sessions
- **Guest Users**: Data is stored locally in JSON file (resets on refresh)

### Security

- Row Level Security (RLS) ensures users can only access their own data
- Passwords are handled securely by Supabase Auth
- Guest mode provides a safe demo experience

## Testing

1. Run the app: `streamlit run app.py`
2. Test each authentication mode:
   - Create a new account
   - Log in with existing credentials
   - Try guest mode
3. Verify data persistence works correctly for each mode

## Troubleshooting

### Common Issues

1. **Supabase connection failed**: Check your URL and anon key in secrets
2. **Table not found**: Run the SQL setup script in Supabase
3. **Authentication errors**: Check Supabase Auth settings
4. **RLS policy errors**: Verify the policies are correctly set up

### Debug Mode

If you encounter issues, the app will show a warning and fall back to guest mode only.

## File Structure

- `app.py`: Main application with authentication integration
- `setup_supabase.sql`: Database setup script
- `.streamlit/secrets.toml.example`: Example secrets configuration
- `requirements.txt`: Python dependencies (includes supabase package)
