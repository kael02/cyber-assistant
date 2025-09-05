#!/usr/bin/env python3
"""
Streamlit UI for Security Query Assistant

A user-friendly web interface for the Security Query Assistant with
advanced memory capabilities, query conversion, and pattern analysis.

Features:
- Interactive chat interface for query conversion
- Multiple output formats (Custom, Splunk, Elastic, etc.)
- Query pattern analysis and insights
- Conversion history and accuracy tracking
- Memory search and exploration
- Query suggestions and recommendations

Requirements:
- pip install streamlit plotly pandas
- Run with: streamlit run streamlit_security_app.py
"""

import json
from datetime import date, datetime
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from diary_assistant import PersonalDiaryAssistant  # Using the security assistant

# Page configuration
st.set_page_config(
    page_title="Security Query Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #E53E3E;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }

    .query-result {
        background-color: #F1F8E9;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #E9ECEF;
        text-align: center;
    }
    .format-badge {
        background-color: #E53E3E;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .success-badge {
        background-color: #38A169;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    .error-badge {
        background-color: #E53E3E;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    .stButton > button {
        background-color: #E53E3E;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .code-block {
        background-color: #2D3748;
        color: #E2E8F0;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "assistant" not in st.session_state:
        st.session_state.assistant = PersonalDiaryAssistant(
            "security_queries_streamlit.db"
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "conversion_history" not in st.session_state:
        st.session_state.conversion_history = []

    if "selected_format" not in st.session_state:
        st.session_state.selected_format = "custom"

    if "correction_mode" not in st.session_state:
        st.session_state.correction_mode = False


def display_chat_message(message, sender):
    """Display a chat message with proper styling."""
    if sender == "user":
        st.markdown(
            f'<div class="chat-message user-message"><strong>You:</strong> {message}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-message assistant-message"><strong>🔍 Assistant:</strong> {message}</div>',
            unsafe_allow_html=True,
        )


def display_query_result(original_query, converted_query, output_format, success=True):
    """Display query conversion results with proper formatting."""
    status_badge = "success-badge" if success else "error-badge"
    status_text = "✅ Success" if success else "❌ Error"
    
    st.markdown(f"**Original Query:** {original_query}")
    st.markdown(f'**Format:** <span class="format-badge">{output_format.upper()}</span> **Status:** <span class="{status_badge}">{status_text}</span>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="query-result"><strong>Converted Query:</strong><br><code>{converted_query}</code></div>',
        unsafe_allow_html=True,
    )


def create_conversion_stats_chart(conversion_data):
    """Create conversion statistics chart."""
    if not conversion_data:
        return None

    df = pd.DataFrame(conversion_data)
    
    # Success rate over time
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_stats = df.groupby('date').agg({
        'success': ['count', 'sum']
    }).reset_index()
    
    daily_stats.columns = ['date', 'total_conversions', 'successful_conversions']
    daily_stats['success_rate'] = (daily_stats['successful_conversions'] / daily_stats['total_conversions']) * 100

    fig = px.line(
        daily_stats,
        x="date",
        y="success_rate",
        title="Query Conversion Success Rate Over Time",
        labels={"success_rate": "Success Rate (%)", "date": "Date"},
    )

    fig.update_layout(yaxis={"range": [0, 100]})
    return fig


def create_format_distribution_chart(conversion_data):
    """Create format usage distribution chart."""
    if not conversion_data:
        return None

    df = pd.DataFrame(conversion_data)
    format_counts = df['format'].value_counts()

    fig = px.pie(
        values=format_counts.values,
        names=format_counts.index,
        title="Output Format Distribution",
    )

    return fig


def create_query_complexity_chart(conversion_data):
    """Create query complexity analysis chart."""
    if not conversion_data:
        return None

    df = pd.DataFrame(conversion_data)
    df['query_length'] = df['original_query'].str.len()
    df['complexity'] = pd.cut(df['query_length'], 
                             bins=[0, 50, 100, 200, float('inf')], 
                             labels=['Simple', 'Medium', 'Complex', 'Very Complex'])

    complexity_counts = df['complexity'].value_counts()

    fig = px.bar(
        x=complexity_counts.index,
        y=complexity_counts.values,
        title="Query Complexity Distribution",
        labels={"x": "Complexity Level", "y": "Number of Queries"},
    )

    return fig


def parse_conversion_record(record_text):
    """Parse conversion record from memory system."""
    try:
        lines = record_text.split('\n')
        record = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                record[key] = value
        
        # Extract relevant fields
        return {
            'timestamp': record.get('time', datetime.now().strftime("%Y-%m-%d %H:%M")),
            'original_query': record.get('input', ''),
            'converted_query': record.get('generated_query', ''),
            'format': record.get('output_format', 'custom'),
            'success': not record.get('status', '').startswith('Failed')
        }
    except:
        return None


def main():
    """Main Streamlit application."""
    initialize_session_state()

    # Header
    st.markdown(
        '<h1 class="main-header">🔍 Security Query Assistant</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Transform natural language into precise security queries with AI-powered memory</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("🔧 Query Converter")

        # Output format selection
        

        # Quick query converter
        quick_query = st.text_area("Enter natural language query:", height=100)
        
        if st.button("🔄 Convert Query"):
            if quick_query.strip():
                converted = st.session_state.assistant.convert_to_query(
                    quick_query, 
                    st.session_state.selected_format
                )
                
                # Add to conversion history
                conversion_record = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'original_query': quick_query,
                    'converted_query': converted,
                    'format': st.session_state.selected_format,
                    'success': not converted.startswith('Error:')
                }
                st.session_state.conversion_history.append(conversion_record)
                
                # Display result
                display_query_result(
                    quick_query, 
                    converted, 
                    st.session_state.selected_format,
                    conversion_record['success']
                )

        # Quick actions
        st.subheader("🎯 Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Daily Summary"):
                summary = st.session_state.assistant.get_conversion_summary()
                st.session_state.chat_history.append(("Daily Summary Request", "user"))
                st.session_state.chat_history.append((summary, "assistant"))

        with col2:
            if st.button("🧠 Analyze Patterns"):
                analysis = st.session_state.assistant.analyze_query_patterns("overall", "month")
                st.session_state.chat_history.append(("Analyze Patterns", "user"))
                st.session_state.chat_history.append((analysis, "assistant"))

        # Query type suggestions
        st.subheader("💡 Get Suggestions")
        suggestion_types = {
            "Network Queries": "network",
            "Process Queries": "process",
            "File System": "file",
            "User Activity": "user",
            "Malware Detection": "malware",
            "General Security": "general"
        }
        
        suggestion_type = st.selectbox("Choose query type:", list(suggestion_types.keys()))
        
        if st.button("💡 Get Suggestions"):
            suggestions = st.session_state.assistant.get_query_suggestions(
                suggestion_types[suggestion_type]
            )
            st.session_state.chat_history.append(
                (f"Get {suggestion_type} suggestions", "user")
            )
            st.session_state.chat_history.append((suggestions, "assistant"))

        # Memory search
        st.subheader("🔍 Memory Search")
        search_query = st.text_input("Search past queries and patterns:")
        if st.button("🔎 Search") and search_query:
            results = st.session_state.assistant.memory_search(search_query)
            st.session_state.chat_history.append((f"Search: {search_query}", "user"))
            st.session_state.chat_history.append((results, "assistant"))

    # Main content area with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["💬 Chat", "📊 Analytics", "📝 Query History", "🔧 Corrections", "📚 Examples"]
    )

    with tab1:
        st.header("💬 Chat with Your Security Assistant")
        st.markdown("Ask questions, convert queries, or get security insights!")

        # Chat interface
        chat_container = st.container()

        with chat_container:
            # Display chat history
            for message, sender in st.session_state.chat_history[-10:]:  # Show last 10 messages
                display_chat_message(message, sender)

        # Chat input
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input("Type your message here...", key="chat_input")
        with col2:
            send_button = st.button("Send 📤")

        if send_button and user_input.strip():
            # Add user message to history
            st.session_state.chat_history.append((user_input, "user"))

            # Get assistant response
            response = st.session_state.assistant.chat_with_memory(user_input)
            st.session_state.chat_history.append((response, "assistant"))

            # Rerun to update the chat display
            st.rerun()

    with tab2:
        st.header("📊 Conversion Analytics")

        # Get conversion statistics from memory
        try:
            memory_results = st.session_state.assistant.memory_search(
                "query conversions statistics success accuracy"
            )
            
            # Today's metrics (mock data for demo)
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Today's Conversions", len([c for c in st.session_state.conversion_history 
                                                   if c['timestamp'].startswith(date.today().strftime("%Y-%m-%d"))]))
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                success_count = len([c for c in st.session_state.conversion_history if c['success']])
                total_count = len(st.session_state.conversion_history)
                success_rate = (success_count / total_count * 100) if total_count > 0 else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")
                st.markdown("</div>", unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                most_used_format = st.session_state.selected_format.upper()
                st.metric("Preferred Format", most_used_format)
                st.markdown("</div>", unsafe_allow_html=True)

            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Queries", total_count)
                st.markdown("</div>", unsafe_allow_html=True)

            # Charts
            if st.session_state.conversion_history:
                col1, col2 = st.columns(2)

                with col1:
                    success_chart = create_conversion_stats_chart(st.session_state.conversion_history)
                    if success_chart:
                        st.plotly_chart(success_chart, use_container_width=True)

                with col2:
                    format_chart = create_format_distribution_chart(st.session_state.conversion_history)
                    if format_chart:
                        st.plotly_chart(format_chart, use_container_width=True)

                # Complexity analysis
                complexity_chart = create_query_complexity_chart(st.session_state.conversion_history)
                if complexity_chart:
                    st.plotly_chart(complexity_chart, use_container_width=True)
            else:
                st.info("📈 Start converting queries to see your analytics!")

        except Exception as e:
            st.error(f"Analytics error: {str(e)}")

    with tab3:
        st.header("📝 Query Conversion History")

        if st.session_state.conversion_history:
            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                date_filter = st.date_input("Filter by date", value=date.today())
            with col2:
                format_filter = st.selectbox("Filter by format", ["All"] + list(format_options.values()))
            with col3:
                status_filter = st.selectbox("Filter by status", ["All", "Success", "Error"])

            # Display conversion history
            filtered_history = st.session_state.conversion_history

            if date_filter:
                filtered_history = [
                    h for h in filtered_history
                    if h['timestamp'].startswith(date_filter.strftime("%Y-%m-%d"))
                ]

            if format_filter != "All":
                filtered_history = [h for h in filtered_history if h['format'] == format_filter]

            if status_filter == "Success":
                filtered_history = [h for h in filtered_history if h['success']]
            elif status_filter == "Error":
                filtered_history = [h for h in filtered_history if not h['success']]

            for i, conversion in enumerate(reversed(filtered_history)):  # Show newest first
                with st.expander(
                    f"{conversion['timestamp']} - {conversion['format'].upper()} - {'✅' if conversion['success'] else '❌'}"
                ):
                    display_query_result(
                        conversion['original_query'],
                        conversion['converted_query'],
                        conversion['format'],
                        conversion['success']
                    )
                    
                    # Correction button
                    if st.button(f"🔧 Correct This Query", key=f"correct_{i}"):
                        st.session_state.correction_mode = True
                        st.session_state.correction_data = conversion
        else:
            st.info("🔍 No conversion history yet. Start by converting your first security query!")

    with tab4:
        st.header("🔧 Query Corrections & Feedback")
        st.markdown("Help improve the assistant by providing corrections to query conversions.")

        # Correction interface
        if st.session_state.get('correction_mode', False):
            correction_data = st.session_state.get('correction_data', {})
            
            st.subheader("Correcting Query Conversion")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Natural Language:**")
                st.code(correction_data.get('original_query', ''), language='text')
                
                st.markdown("**Generated Query:**")
                st.code(correction_data.get('converted_query', ''), language='sql')
            
            with col2:
                corrected_query = st.text_area("Enter the corrected query:", height=150)
                feedback = st.text_area("Additional feedback (optional):", height=100)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("✅ Submit Correction"):
                        if corrected_query.strip():
                            result = st.session_state.assistant.record_correction(
                                correction_data.get('original_query', ''),
                                correction_data.get('converted_query', ''),
                                corrected_query,
                                correction_data.get('format', 'custom'),
                                feedback
                            )
                            st.success(result)
                            st.session_state.correction_mode = False
                            st.rerun()
                
                with col_b:
                    if st.button("❌ Cancel"):
                        st.session_state.correction_mode = False
                        st.rerun()
        else:
            st.markdown("Select a query from the history tab to provide corrections.")
            
            # General feedback form
            st.subheader("General Feedback")
            general_feedback = st.text_area("Share your thoughts on the assistant's performance:")
            
            if st.button("📝 Submit Feedback") and general_feedback:
                # Record general feedback
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
                feedback_record = f"""
                Time: {current_time}
                Type: General Feedback
                Feedback: {general_feedback}
                """
                
                st.session_state.assistant.memory_system.record_conversation(
                    user_input="General feedback submission",
                    ai_output=feedback_record
                )
                st.success("Thank you for your feedback!")

    with tab5:
        st.header("📚 Query Examples & Best Practices")
        
        # Example queries by category
        examples = {
            "Network Security": [
                {
                    "natural": "Find all network connections from IP address 192.168.1.100",
                    "custom": "src_ip:192.168.1.100",
                    "splunk": 'src_ip="192.168.1.100"',
                    "elastic": 'source.ip:"192.168.1.100"'
                },
                {
                    "natural": "Show me all failed login attempts in the last 24 hours",
                    "custom": "event_type:login AND status:failed AND @timestamp:[now-24h TO now]",
                    "splunk": 'eventtype="login" status="failed" earliest=-24h',
                    "elastic": 'event.type:"login" AND event.outcome:"failure" AND @timestamp:[now-24h TO now]'
                }
            ],
            "Process Monitoring": [
                {
                    "natural": "Find all PowerShell processes running with admin privileges",
                    "custom": "process_name:powershell* AND privileges:admin",
                    "splunk": 'process="powershell*" privileges="admin"',
                    "elastic": 'process.name:powershell* AND user.roles:admin'
                }
            ],
            "File System": [
                {
                    "natural": "Detect file modifications in system directories",
                    "custom": "file_path:/System/* AND action:modify",
                    "splunk": 'file_path="/System/*" action="modify"',
                    "elastic": 'file.path:/System/* AND event.action:modify'
                }
            ]
        }

        # Display examples
        for category, category_examples in examples.items():
            st.subheader(f"🔒 {category}")
            
            for i, example in enumerate(category_examples):
                with st.expander(f"Example {i+1}: {example['natural']}"):
                    st.markdown("**Natural Language Query:**")
                    st.code(example['natural'], language='text')
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**Custom Format:**")
                        st.code(example['custom'], language='text')
                    with col2:
                        st.markdown("**Splunk Format:**")
                        st.code(example['splunk'], language='text')
                    with col3:
                        st.markdown("**Elastic Format:**")
                        st.code(example['elastic'], language='text')
                    
                    # Try this example button
                    if st.button(f"🔄 Try This Example", key=f"example_{category}_{i}"):
                        converted = st.session_state.assistant.convert_to_query(
                            example['natural'],
                            st.session_state.selected_format
                        )
                        st.success("Example converted! Check the result above.")

        # Best practices
        st.subheader("💡 Best Practices for Query Conversion")
        
        best_practices = [
            "**Be specific**: Include exact field names, IP addresses, or process names when known",
            "**Use time ranges**: Specify time windows for better performance (e.g., 'in the last hour')",
            "**Include context**: Mention the type of security event you're looking for",
            "**Use operators**: Learn common operators like AND, OR, NOT for complex queries",
            "**Test queries**: Start simple and add complexity gradually"
        ]
        
        for practice in best_practices:
            st.markdown(f"• {practice}")

    # Footer
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🔍 **Security Query Assistant** - Convert natural language to security queries")
    with col2:
        st.markdown("🧠 **Powered by Advanced Memory** - Learning from every conversion")


if __name__ == "__main__":
    main()