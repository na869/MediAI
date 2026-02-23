from flask import Flask, render_template, request, session, flash, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, Email, EqualTo
# Import necessary libraries
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Add a secret key for session management

class SignupForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

# Define the dataset with text and corresponding labels
dataset = [("Open discussion. Between the Transfer Portal and the NIL, will the @NCAA become obsolete as an organization and governing body? @zlancaster91 @RAllenGoPokes #Hopelessness #GoPokes #LoyalandTrue", "Depression"),
    ("Plenty of things are changing in my life and the lives of those around me. There is one thing that doesn't change, my #hopelessness.", "Depression"),
    ("I feel a little hopeless. Anyone else? #hopelessness", "Depression"),
    ("Which is more healthy? Hope, or hopelessness? #hope #Hopelessness #Mentalhealth", "Depression"),
    ("\"So someone tell me how do I get over #HOPELESSNESS? I live in a world of #poverty surrounded by #PoorPeople. People help us that are not much better off than we are. If not for my son's inability to care for himself I would likely give into the Hopelessness &amp; darkness.\"", "Depression"),
    ("No parent deserves to experience the Indian legal system. #hopelessness", "Depression"),
    ("\"Being in a #union also looks a lot like being #alone. It can feel like thereâ€™s no worse place to be sometimes. #hopelessness #unions #workersrights\"", "Depression"),
    ("\"I am so glad that @GreysABC is tackling the huge #healthcare professionals shortage we are facing and so saddened that we are not really living in their post pandemic world. The #burnout, the #hopelessness, the #stress are palpable at every level and department of the system.\"", "Depression"),
    ("If you know someone whoâ€™s depressed please resolve to never ask them why. #depression isnâ€™t a straightforward response to a bad situation; depression just is, like the weather. Try to understand the blackness, lethargy, #hopelessness and loneliness theyâ€™re going through.", "Depression"),
    ("#apprentice. And @wossy steals the best line of the show re Gordon brown", "Positive"),
    ("#asot 400 is really really amazing. i can't stop listening do you like asot &amp;armin ??", "Positive"),
    ("#asot400 after this weekend, i hate twitter", "Positive"),
    ("#asot400 AIR FOR LIFE... so many hours spent with this song learning how to use loops.", "Positive"),
    ("#ASOT400 Buenos Aires @Argentina, under A state of trance celebration armin.", "Positive"),
    ("#ASOT400 I think Marcus Shultz melted some face. I gotta admit, this song is melting mine", "Positive"),
    ("#asot400 thanks you arminnnn", "Positive"),
    ("#asot400 Yes they are. Episode 001 now", "Positive"),
    ("#assassinate is also trending because #spymaster lets you assassinate time", "Positive"),
    ("#asylm is showing up on trending topics for me now!!", "Positive"),
    ("#asylm Jared is awesome and soo so so cute", "Positive"),
    ("#asylm Misha photo op... He still remembers me. He is so adorable!", "Positive"),
    ("#asylm Thanks to all the people who tweeted! XD", "Positive"),
    ("#babyupdate I've been sitting holding Callum. He's doing great, feeding well. They're gradually lowering the drip as he adjusts himself.", "Positive"),
    ("#bachelorette Uh, my husband watching Bachelorette and I'm watching the OK State/Clemson baseball game", "Positive"),
    ("\"Good Morning to everyone except @TheUSFWizard very upsetting to see the election results last night across the country. The Racist Variant of Trumpism has arrived. I really worry for the country. Hereâ€™s to hoping USF Football can get my mind off it Saturday. #Depressed\"", "Anorexia"),
    ("\"I'm envious of people having good siblings #Sad #depressed\"", "Anorexia"),
    ("\"saw my dog blowing my girls back out today #depressed #foreveralone #cheatedon #whyme\"", "Anorexia"),
    ("I need to stop bullshitting wit my life so I can gtfo ðŸ˜¤ #depressed", "Anorexia"),
    ("I know I'm #depressed when I can't even listen to music...", "Anorexia"),
    ("\"My source of stress is I have 10th - 89% 12th - 59.08% B.E. - 77.14 and irrespective of my knowledge and my passion to work with @Wipro, I got rejected from @WiproCareers on the basis of my 12th marks. #single_piece_of_paper_can_decide_our_future #depressed\"", "Anorexia"),
    ("\"Please always check on some of your friends! Some of us seems to be okay but we ain't ðŸ˜¢ #depression #NFTdrops #depressed #pain #dying #help\"", "Anorexia"),
    ("I havent felt like myself in over a year #depressed", "Anorexia"),
    ("\"I'm really struggling with my new job. It's been a month now, I don't cry as much, but I feel hollow. The benefits are great, but I can't shake the feeling that something is wrong. #Job #depressed #working\"", "Anorexia"),
    ("\"#truth The strain on our global community is having an impact on my #mentalhealth. Things are ok for me. I feel like I have no right to be #depressed. This is a pointless thought tho. It doesnâ€™t matter if others have it worse. Iâ€™m depressed and I can deal with it or not.\"", "Anorexia"),
    ("\"The amount of ðŸ¦‡ðŸ’© crazy that won/moved to a runoff across Miami-Dade yesterday is mind-boggling. The voter turnout was pathetic. #depressed #ElectionDay2021\"", "Anorexia"),
    ("I love feeling like shitâœŒðŸ»#Imfine #Depressed", "Anorexia"),
    ("|| at 190 and get no interactions #emo #emoboy #depressed ðŸ˜­ðŸ˜­ðŸ˜­ðŸ˜­", "Anorexia"),
    ("On a scale of 1-#depressed, how funny are you?", "Anorexia"),
    ("jay didnâ€™t share his chicken alfredo with me #depressed", "Anorexia"),
    ("\"Night Prayer: Kyrie eleison down the road that I must travel. #Pray 4 #Iran #India #Mexico #Ecuador #TigrayFamine #Venezuela #Colombia #Cuba #Lebanon #Chile #Japan #Afghanistan #Haiti #Lonely #depressed #poor #Jesus #Peace #love #help\"", "Anorexia"),
    ("\"Sometimes I be in my feelings and all of sudden I be like â€œfuck being sad bro, fuck whoever or whatever is holding you from being happyâ€ and like fr nothing is worth our happiness. Be happy and stay positive even if shit gets hard #happy #depressed #strong\"", "Anorexia"),
    ("remember when i used to throw up everyday i was so #depressed", "Anorexia"),
    ("Damn, yâ€™all I just raided @bearbubb and he called my raid fakeðŸ˜­ #depressed #depressoexpresso #speakingfromtheheartbfletcher", "Anorexia"),
    ("\"this\"", "Unknown"),
    ("\"I think about this quote often: \"\"When nobody wakes you up in the morning, and when nobody waits for you at night, and when you can do whatever you want. What do you call it, freedom or loneliness?\"\" - Charles Bukowski #Freedom #loneliness\"", "Anorexia"),
    ("If you've never dealt with #depression, for me at least, it's like I have the flu without the cough, with the addition of fog brain, emotional waves, intolerability, lethargy, guilt, #loneliness, all bundled up into a ball of #fuck. It can come in waves, or last for months+", "Anorexia"),
    ("\"Loneliness is not when you don't have loved ones in life, it's not having someone in your life who truly gets you. #Loneliness #writer #writerslift #WritingCommunity\"", "Anorexia"),
    ("\"I donâ€™t want to be the center of the room but Iâ€™d at least like to be invited to the roomâ€¦ ðŸ˜’ðŸ¤·ðŸ»â€â™€ï¸ #DepressionTalk #loneliness #aloneinaseaoffaces\"", "Anorexia"),
    ("\"Yes. Having an extremely difficult time coping with life at the moment. Lol. It is so hilarious. #bipolar #depression #loneliness\"", "Anorexia"),
    ("Gr8 wealth, miracle works, peace, health, grace, wisdom, success &amp; prospering opportunities come 2 me n unprecedented &amp; unexpected ways! Iâ€™m a daily recipient of them all thru 202WON #Day290 #MyBirthday #OctoberOvertures #loneliness #humble #plenty #yearoffruition #unstoppable", "Anorexia"),
    ("We always think of \"loneliness\" as a state of \"only oneself\", but in fact, even if there are people, they still feel this way when \"no one helps oneself\" or \"no one understands oneself\".#loneliness #Lonely", "Anorexia"),
    ("\"The sky is beautiful tonight. Clear. And bright. The wind is downright cold. It was a shyte day. And will be shyte for a bit longer. Having people over for supper while having an episode is always a challenge. Oh well, fuck. #bipolar #depression #loneliness\"", "Anorexia"),
    ("\"Night Prayer: Kyrie eleison down the road that I must travel. #Pray 4 #Iran #India #Mexico #Ecuador #TigrayFamine #Venezuela #Colombia #Cuba #Lebanon #Chile #Japan #Afghanistan #Haiti #Lonely #depressed #poor #Jesus #Peace #love #help\"", "Anorexia"),
    ("\"Sometimes I be in my feelings and all of sudden I be like â€œfuck being sad bro, fuck whoever or whatever is holding you from being happyâ€ and like fr nothing is worth our happiness. Be happy and stay positive even if shit gets hard #happy #depressed #strong\"", "Anorexia"),
    ("remember when i used to throw up everyday i was so #depressed", "Anorexia"),
    ("Damn, yâ€™all I just raided @bearbubb and he called my raid fakeðŸ˜­ #depressed #depressoexpresso #speakingfromtheheartbfletcher", "Anorexia"),
    ("\"I'm envious of people having good siblings #Sad #depressed\"", "Anorexia"),
    ("\"saw my dog blowing my girls back out today #depressed #foreveralone #cheatedon #whyme\"", "Anorexia"),
    ("I need to stop bullshitting wit my life so I can gtfo ðŸ˜¤ #depressed", "Anorexia"),
    ("I know I'm #depressed when I can't even listen to music...", "Anorexia"),
    ("\"My source of stress is I have 10th - 89% 12th - 59.08% B.E. - 77.14 and irrespective of my knowledge and my passion to work with @Wipro, I got rejected from @WiproCareers on the basis of my 12th marks. #single_piece_of_paper_can_decide_our_future #depressed\"", "Anorexia"),
    ("\"Please always check on some of your friends! Some of us seems to be okay but we ain't ðŸ˜¢ #depression #NFTdrops #depressed #pain #dying #help\"", "Anorexia"),
    ("I havent felt like myself in over a year #depressed", "Anorexia"),
    ("\"I'm really struggling with my new job. It's been a month now, I don't cry as much, but I feel hollow. The benefits are great, but I can't shake the feeling that something is wrong. #Job #depressed #working\"", "Anorexia"),
    ("\"#truth The strain on our global community is having an impact on my #mentalhealth. Things are ok for me. I feel like I have no right to be #depressed. This is a pointless thought tho. It doesnâ€™t matter if others have it worse. Iâ€™m depressed and I can deal with it or not.\"", "Anorexia"),
    ("\"The amount of ðŸ¦‡ðŸ’© crazy that won/moved to a runoff across Miami-Dade yesterday is mind-boggling. The voter turnout was pathetic. #depressed #ElectionDay2021\"", "Anorexia"),
    ("I love feeling like shitâœŒðŸ»#Imfine #Depressed", "Anorexia"),
    ("|| at 190 and get no interactions #emo #emoboy #depressed ðŸ˜­ðŸ˜­ðŸ˜­ðŸ˜­", "Anorexia"),
    ("On a scale of 1-#depressed, how funny are you?", "Anorexia"),
    ("jay didnâ€™t share his chicken alfredo with me #depressed", "Anorexia"),
    ("\"Night Prayer: Kyrie eleison down the road that I must travel. #Pray 4 #Iran #India #Mexico #Ecuador #TigrayFamine #Venezuela #Colombia #Cuba #Lebanon #Chile #Japan #Afghanistan #Haiti #Lonely #depressed #poor #Jesus #Peace #love #help\"", "Anorexia"),
    ("\"Sometimes I be in my feelings and all of sudden I be like â€œfuck being sad bro, fuck whoever or whatever is holding you from being happyâ€ and like fr nothing is worth our happiness. Be happy and stay positive even if shit gets hard #happy #depressed #strong\"", "Anorexia"),
    ("remember when i used to throw up everyday i was so #depressed", "Anorexia"),
    ("Damn, yâ€™all I just raided @bearbubb and he called my raid fakeðŸ˜­ #depressed #depressoexpresso #speakingfromtheheartbfletcher", "Anorexia"),
    ("\"this\"", "Unknown"),
    ("Open discussion. Between the Transfer Portal and the NIL, will the @NCAA become obsolete as an organization and governing body? @zlancaster91 @RAllenGoPokes #Hopelessness #GoPokes #LoyalandTrue","Depression"),
    ("Plenty of things are changing in my life and the lives of those around me. There is one thing that doesn't change, my #hopelessness.
","Depression")

   ]

# Define a function to predict the label based on the input text
def predict_label(input_text):
    for text, label in dataset:
        if input_text.lower() in text.lower():
            return label
    return "Unknown"

@app.route('/')
def home():
    return render_template('home.html', title="Home")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Hardcoded valid username and password
        valid_username = 'admin'
        valid_password = 'admin'
        
        username = request.form['username']
        password = request.form['password']
        if username == valid_username and password == valid_password:
            session['logged_in'] = True
            flash('Login successful!', 'success')
            return redirect(url_for('home'))  # Redirect to the home page
        else:
            flash('Invalid username or password', 'error')
            return render_template('login.html', title="Login")
    else:
        return render_template('login.html', title="Login")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password = form.password.data
        
        # Store user information (you may store it in a database instead)
        session['username'] = username
        session['email'] = email
        flash('Signup successful! Please login.', 'success')
        return redirect(url_for('signup_success'))  # Redirect to the signup success page
    
    return render_template('signup.html', title="Signup", form=form)

@app.route('/signup_success')
def signup_success():
    return render_template('signup_success.html', title="Signup Success")

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('logout_success'))

@app.route('/logout_success')
def logout_success():
    return render_template('logout.html')

@app.route("/Admin", methods=["GET", "POST"])
def Admin():
    if request.method == "POST":
        input_text = request.form["input_text"]
        predicted_label = predict_label(input_text)
        return render_template("index.html", input_text=input_text, predicted_label=predicted_label)
    return render_template("index.html")
# Load dataset
df = pd.read_csv("depression_detection.csv")


# Initialize Dash app
dash_app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define color options for bars
color_options = [
    {"label": "Blue", "value": "blue"},
    {"label": "Red", "value": "red"},
    {"label": "Green", "value": "green"},
    {"label": "Yellow", "value": "yellow"},
    {"label": "Orange", "value": "orange"},
    {"label": "Standard", "value": "standard"},  # Standard color option
    {"label": "Custom", "value": "custom"}  # Custom color option
]

# Define layout of the dashboard
dash_app.layout = html.Div([
    html.H1("Interactive Dashboard", style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    dbc.Row([
        dbc.Col([
            # Dropdown for selecting columns for Chart 1
            html.Label("Select Column for Chart 1:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id="column-dropdown-1",
                options=[{'label': col, 'value': col} for col in df.columns],
                value='Age',  # Default value
                style={'width': '100%'}
            ),
            # Dropdown for selecting chart type for Chart 1
            html.Label("Select Chart Type for Chart 1:", style={'fontWeight': 'bold', 'marginTop': '20px'}),
            dcc.Dropdown(
                id="chart-type-dropdown-1",
                options=[
                    {'label': 'Bar Chart', 'value': 'bar'},
                    {'label': 'Pie Chart', 'value': 'pie'},
                    {'label': 'Histogram', 'value': 'histogram'},
                    {'label': 'Column Chart', 'value': 'column'}
                ],
                value='bar',  # Default value
                style={'width': '100%'}
            ),
            # Dropdown for selecting bar color for Chart 1
            html.Label("Select Bar Color for Chart 1:", style={'fontWeight': 'bold', 'marginTop': '20px'}),
            dcc.Dropdown(
                id="bar-color-dropdown-1",
                options=color_options,  # Use the color options defined above
                value='blue',  # Default value
                style={'width': '100%'}
            ),
            # Chart 1
            dcc.Graph(id='chart-1'),
        ], width=6),
        
        dbc.Col([
            # Dropdown for selecting columns for Chart 2
            html.Label("Select Column for Chart 2:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id="column-dropdown-2",
                options=[{'label': col, 'value': col} for col in df.columns],
                value='label',  # Default value
                style={'width': '100%'}
            ),
            # Dropdown for selecting chart type for Chart 2
            html.Label("Select Chart Type for Chart 2:", style={'fontWeight': 'bold', 'marginTop': '20px'}),
            dcc.Dropdown(
                id="chart-type-dropdown-2",
                options=[
                    {'label': 'Bar Chart', 'value': 'bar'},
                    {'label': 'Pie Chart', 'value': 'pie'},
                    {'label': 'Histogram', 'value': 'histogram'},
                    {'label': 'Column Chart', 'value': 'column'}
                ],
                value='pie',  # Default value
                style={'width': '100%'}
            ),
            # Dropdown for selecting bar color for Chart 2
            html.Label("Select Bar Color for Chart 2:", style={'fontWeight': 'bold', 'marginTop': '20px'}),
            dcc.Dropdown(
                id="bar-color-dropdown-2",
                options=color_options,  # Use the color options defined above
                value='blue',  # Default value
                style={'width': '100%'}
            ),
            # Chart 2
            dcc.Graph(id='chart-2'),
        ], width=6),
    ], style={'margin': 'auto', 'maxWidth': '1200px'}),
], style={'padding': '20px'})

# Define callback to update all charts based on selected column, chart type, and bar color
@dash_app.callback(
    [Output('chart-1', 'figure'),
     Output('chart-2', 'figure')],
    [Input('column-dropdown-1', 'value'),
     Input('chart-type-dropdown-1', 'value'),
     Input('bar-color-dropdown-1', 'value'),
     Input('column-dropdown-2', 'value'),
     Input('chart-type-dropdown-2', 'value'),
     Input('bar-color-dropdown-2', 'value')]
)
def update_charts(column_1, chart_type_1, bar_color_1, column_2, chart_type_2, bar_color_2):
    fig1 = update_chart(column_1, chart_type_1, bar_color_1)
    fig2 = update_chart(column_2, chart_type_2, bar_color_2)
    return fig1, fig2

# Function to update chart based on selected column, chart type, and bar color
def update_chart(selected_column, chart_type, bar_color):
    if chart_type == 'bar':
        fig = px.bar(df, x='text', y=selected_column, title=f'{selected_column} Distribution')
        if bar_color != 'custom':
            fig.update_traces(marker_color=bar_color)
    elif chart_type == 'pie':
        fig = px.pie(df, names=selected_column, title=f'{selected_column} Distribution')
    elif chart_type == 'histogram':
        fig = px.histogram(df, x=selected_column, title=f'{selected_column} Distribution')
    else:  # column chart
        fig = px.scatter(df, x='text', y=selected_column, title=f'{selected_column} Distribution')
    return fig

# Define route for the Dash app
@dash_app.server.route('/interactivedashboard', methods=['GET', 'POST'])
def interactivedashboard():
    return dash_app.home()



if __name__ == '__main__':
    app.run(debug=True)  # Change port to 8051 (or any available port)
