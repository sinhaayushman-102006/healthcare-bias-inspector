import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_bias(acc_dict, title):
    df = pd.DataFrame({
        "Group": list(acc_dict.keys()),
        "Accuracy": list(acc_dict.values())
    })
    
    fig = px.bar(
        df, 
        x="Group", 
        y="Accuracy",
        title=title,
        color="Group",
        text_auto=".2f",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        title_font=dict(size=20, color="#1e293b", family="Outfit, sans-serif"),
        showlegend=False,
        yaxis=dict(range=[0, 1]),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_traces(textposition="outside", textfont_size=14)
    return fig

def plot_comparison(before, after):
    labels = list(before.keys())
    before_vals = list(before.values())
    after_vals = list(after.values())

    fig = go.Figure(data=[
        go.Bar(name='Before Fix', x=labels, y=before_vals, text=[f"{v:.2f}" for v in before_vals], textposition="outside", marker_color='#94a3b8'),
        go.Bar(name='After Fix', x=labels, y=after_vals, text=[f"{v:.2f}" for v in after_vals], textposition="outside", marker_color='#3b82f6')
    ])
    
    fig.update_layout(
        title="Accuracy Comparison Before and After Mitigation",
        title_font=dict(size=20, color="#1e293b", family="Outfit, sans-serif"),
        barmode='group',
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
