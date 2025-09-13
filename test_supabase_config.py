#!/usr/bin/env python3
"""
Test script to verify Supabase configuration and Google OAuth setup.
Run this script to check if your Supabase project is properly configured.
"""

import os
import sys
from supabase import create_client, Client

def test_supabase_config():
    """Test Supabase configuration and Google OAuth setup."""
    print("🔍 Testing Supabase Configuration...")
    print("=" * 50)
    
    # Check if secrets are available
    try:
        import streamlit as st
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_ANON_KEY"]
        print("✅ Streamlit secrets found")
    except Exception as e:
        print(f"❌ Streamlit secrets not found: {e}")
        print("💡 Make sure you have a .streamlit/secrets.toml file with:")
        print("   SUPABASE_URL = 'your-project-url'")
        print("   SUPABASE_ANON_KEY = 'your-anon-key'")
        return False
    
    # Test Supabase connection
    try:
        supabase: Client = create_client(url, key)
        print("✅ Supabase client created successfully")
    except Exception as e:
        print(f"❌ Failed to create Supabase client: {e}")
        return False
    
    # Test database connection
    try:
        response = supabase.table("weight_logs").select("*").limit(1).execute()
        print("✅ Database connection successful")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("💡 Make sure you've run the setup_supabase.sql script")
        return False
    
    # Test Google OAuth URL generation
    try:
        redirect_url = "https://gym-monster.streamlit.app"  # Update with your URL
        auth_url = f"{url}/auth/v1/authorize?provider=google&redirect_to={redirect_url}"
        print("✅ Google OAuth URL generated successfully")
        print(f"   OAuth URL: {auth_url}")
    except Exception as e:
        print(f"❌ Failed to generate OAuth URL: {e}")
        return False
    
    print("\n🎯 Next Steps:")
    print("1. Go to your Supabase dashboard")
    print("2. Navigate to Authentication → Providers")
    print("3. Enable the Google provider")
    print("4. Add your Google OAuth credentials")
    print("5. Set redirect URL to:", redirect_url)
    
    return True

if __name__ == "__main__":
    success = test_supabase_config()
    if success:
        print("\n✅ Configuration test completed successfully!")
    else:
        print("\n❌ Configuration test failed. Please fix the issues above.")
        sys.exit(1)
