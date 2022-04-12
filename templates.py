# -*- coding: utf-8 -*-

import inspect
import json
import re

from code_gen import iteratively_request_code


def get_comment(language, ml=True):
    '''Get the comment symbol for the specified programming language'''
    if language.lower() == 'python':
        if ml:
            return ('"""', '"""')
        else:
            return '#'
    elif language.lower() == 'matlab':
        if ml:
            return ('%{', '%}')
        else:
            return '%'
    else:
        if ml:
            return ('/*', '*/')
        else:
            return '//'


def get_api_template(api_name, task, params, token=None):
    # few shot learning
    template = '"""Send a request to the Open Notify API '
    template += 'to GET an estimate for when the ISS will fly over a specified point.\n'''
    template += 'Use the following parameters:\n'
    template += '''lat: "45"
lon: "180"\n"""\n'''
    template += '''query = {'lat':'45', 'lon':'180'}
response = requests.get("http://api.open-notify.org/iss-pass.json", params=query)
print(response.json())\n\n'''
    template += '"""Send a request to the google maps API to get the geocordinates of a location.\n'
    template += 'Use the following parameters:\n'
    template += 'address: "1600 Amphitheatre Parkway, Mountain View, CA"\n"""\n'''
    template += '''query = {'address':'1600 Amphitheatre Parkway, Mountain View, CA'}
response = requests.get("http://maps.googleapis.com/maps/api/geocode/json", params=query)
print(response.json())\n\n'''
    # user provided parameters'
    template += f'"""Send a request to the {api_name} API to {task}.\n'
    template += 'Use the following parameters:\n'
    if not isinstance(params, str):
        params = json.dumps(params)
    template += params + "\n"
    if token:
        template += f"Use the following API token header:\n{token}\n"
    template += '"""\nquery ='
    return template


def get_api_request_code(api_name, task, params, token=None):
    prompt = get_api_template(api_name, task, params, token)
    code = iteratively_request_code(prompt, max_tokens=128, temperature=0.2,
                                    presence_penalty=0.3, frequency_penalty=0.4,
                                    stop=['"""', '\n\n\n'])
    return code


def get_sql_explanation_template(query):
    template = """SELECT DISTINCT department.name
    FROM department
    JOIN employee ON department.id = employee.department_id
    JOIN salary_payments ON employee.id = salary_payments.employee_id
    WHERE salary_payments.date BETWEEN '2020-06-01' AND '2020-06-30'
    GROUP BY department.name
    HAVING COUNT(employee.id) > 10;
    -- Explanation of the above query in human readable format
    -- Select the name of each department that has more than 10 employees
    -- who were paid between June 1st and June 30th
    -- Join the tables together
    -- Group by the department name
    -- Having the count of employees in each department is greater than 10

    SELECT * FROM albums
    WHERE ArtistId = (SELECT ArtistId FROM artists WHERE Name = 'Adele')
    AND Title = 'Hello World'
    -- Explanation of the above query in human readable format
    -- Select all the albums that are by the artist 'Adele'
    -- and are titled 'Hello World'
    
    <query>
    -- Explanation of the above query in human readable format
    --"""
    template = inspect.cleandoc(template)
    # preprocess the query
    query = query.strip()
    query = inspect.cleandoc(query)
    template = re.sub(r"<query>", query, template)
    return template


def sql2nl(query):
    prompt = get_sql_explanation_template(query)
    explanation = iteratively_request_code(prompt, max_tokens=256,
                                           temperature=0.4,
                                           stop=["#", "\n\n", "SELECT", '"""'])
    explanation = "--" + explanation
    # remove lines without --
    explanation_split = explanation.split('\n')
    new_explanation = []
    # to prevent duplicates
    new_explanation_set = set()
    for line in explanation_split:
        if line.startswith('--') and line not in new_explanation_set:
            new_explanation.append(line)
            new_explanation_set.add(line)
        else:
            break
    explanation = "\n".join(new_explanation)
    return explanation


def get_sql_generation_template(table_names, col_names, task, sql_engine="MySQL"):
    if len(table_names) < 1:
        print("ERROR: table_names must contain atleast one table")
        return ''
    if len(table_names) != len(col_names):
        print("ERROR: len(table_names) must equal len(col_names)")
        return ''
    template = '''"""
    Table albums, columns = [AlbumId, Title, ArtistId]
    Table artists, columns = [ArtistId, Name]
    Table media_types, columns = [MediaTypeId, Name]
    Table playlists, columns = [PlaylistId, Name]
    Table playlist_track, columns = [PlaylistId, TrackId]
    Table tracks, columns = [TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice]

    Create a MySQL query for all albums by Adele from her album called Hello World"""
    query = """SELECT * FROM albums
                JOIN artists ON albums.ArtistId = artists.ArtistId
                WHERE artists.Name = 'Adele' AND albums.Title = 'Hello World'"""
    
    """'''
    template = inspect.cleandoc(template)
    # preprocess the query
    for i, table_name in enumerate(table_names):
        cols = col_names[i]
        template += f'Table {table_name}, columns = {cols}\n'
    template += f'\nCreate a {sql_engine} query to {task}\n'
    template += '"""\nquery = """SELECT'
    return template


def nl2sql(table_names, col_names, task, sql_engine="MySQL"):
    prompt = get_sql_generation_template(
        table_names, col_names, task, sql_engine)
    sql = iteratively_request_code(prompt, max_tokens=256, temperature=0.1,
                                   stop=['#', '\n\n', ';', '"""'])
    sql = "SELECT" + sql
    return sql


def get_code2nl_template(code):
    prompt = '''class Log:
    def __init__(self, path):
        dirname = os.path.dirname(path)
        os.makedirs(dirname, exist_ok=True)
        f = open(path, "a+")

        # Check that the file is newline-terminated
        size = os.path.getsize(path)
        if size > 0:
            f.seek(size - 1)
            end = f.read(1)
            if end != "\\n":
                f.write("\\n")
        self.f = f
        self.path = path

    def log(self, event):
        event["_event_id"] = str(uuid.uuid4())
        json.dump(event, self.f)
        self.f.write("\\n")

    def state(self):
        state = {"complete": set(), "last": None}
        for line in open(self.path):
            event = json.loads(line)
            if event["type"] == "submit" and event["success"]:
                state["complete"].add(event["id"])
                state["last"] = event
        return state'''

    exp = '''\n\n"""
Here's what the above code is doing:
1. The constructor creates a directory for the log file if it doesn't exist.
2. The log() method writes a JSON-encoded event to the log file.
3. The state() method returns a dictionary with the set of complete tasks and the most recent event.
"""'''
    prompt += exp
    prompt += '''\n
<code>

"""Here's what the above code is doing:
1.'''

    prompt = re.sub("<code>", code, prompt)
    return prompt


def code2nl(code):
    prompt = get_code2nl_template(code)
    code = iteratively_request_code(
        prompt, temperature=0.3, max_tokens=128, frequency_penalty=0.2, stop=['"""', '*/', '#'])
    code = "1." + code
    return code


def get_error_explanation_template(function, context='', error=''):
    if context != '':
        template = context + "\n\n"
        template += function + "\n\n"
    else:
        template = function + "\n\n"
    template += '"""The function above does not work as intended.\n'
    if error != '':
        template += f'Here is the error message:\n{error}\n'
    template += 'The following corrections are needed:\n1.'
    return template


def get_error_explanation(function, context='', error=''):
    prompt = get_error_explanation_template(function, context, error)
    code = iteratively_request_code(prompt, temperature=0.2, stop=['#', '"""', '//', '/*'],
                                    max_tokens=256, frequency_penalty=1)
    return code


def get_fix_bugs_template(function, language, get_fn_header=False, context=''):
    function = function.strip()
    parts = function.split('\n')
    header = parts[0]
    body = parts[1:]
    docstring = None
    if '"""' in body[0]:
        docstring = body[0] + '\n'
        i = 1
        while '"""' not in body[i]:
            docstring += body[i] + '\n'
            i += 1
        docstring += body[i] + '\n'
    cmt_start, cmt_end = get_comment(language)
    if context != '':
        template = context + "\n\n"
        template += '"""Fix bugs in the below code\n\n"""'
    else:
        template = '"""Fix bugs in the below code\n\n"""'
    template += '''import Random
a = random.randint(1,12)
b = random.randint(1,12)
for i in range(10):
    question = "What is "+a+" x "+b+"? "
    answer = input(question)
    if answer = a*b
        print (Well done!)
    else:
        print("No.")
    
"""Fixed Code"""
import random
a = random.randint(1,12)
b = random.randint(1,12)
for i in range(10):
    question = "What is "+str(a)+" x "+str(b)+"? "
    answer = input(question)
    if answer == str(a*b):
        print ("Well done!")
    else:
        print("No.")\n\n'''

    template = f'{cmt_start}Fix bugs in the below code{cmt_end}\n'
    template += function + "\n\n"
    template += f'{cmt_start}Fixed Code{cmt_end}\n'
    template += header
    if docstring:
        template += f"\n{docstring}"
    if get_fn_header:
        return template, header
    else:
        return template


def fix_bugs(function, language, context=''):
    function = function.strip()
    if 'python' in language.lower():
        stop = ['"""', '\n\n', '###']
        if '#' not in function:
            stop.append('#')
        prompt = "# Python 3\n"
    else:
        stop = ['#', '"""', '/*']
        if '//' not in function:
            stop.append('//')
        prompt = f'# {language}\n'
    template = get_fix_bugs_template(function, language, True, context)
    prompt += template[0]
    fn_header = template[1]
    temperature = 0
    code = iteratively_request_code(prompt, max_tokens=256, frequency_penalty=0.4,
                                    temperature=temperature, stop=stop)
    while code.strip() == '':
        temperature += 0.1
        code = iteratively_request_code(prompt, max_tokens=256, frequency_penalty=0.4,
                                        temperature=temperature, stop=stop)
    fixed_code = fn_header + "\n" + code
    return fixed_code


def get_code2docstring_template(code):
    template = '''def add_binary(a, b):
    binary_sum = bin(a+b)[2:]
    return binary_sum
    
# Summary of above function with an elaborate, high quality docstring:
"""
Returns the sum of two decimal numbers in binary digits.
    Parameters:
        a(int): A decimal integer
        b(int): Another decimal integer

    Returns:
        binary_sum (str): Binary string of the sum of a and b
"""\n\n'''
    template += '''class Person:

    def __init__(self, name, surname, age):
        self.name = name
        self.surname = surname
        self.age = age

    def info(self, additional=""):
        print(f'My name is {self.name} {self.surname}. I am {self.age} years old.' + additional)
    
# Summary of above class with an elaborate, high quality docstring:
"""
    A class to represent a person.

    Attributes
    name: str
        first name of the person
    surname: str
        family name of the person
    age: int
        age of the person

    Methods
    info(additional=""):
        Prints the person's name and age.
"""\n\n'''
    prompt = '''<code>

# Summary of above code with an elaborate, high quality docstring:
"""'''
    prompt = inspect.cleandoc(prompt)
    return re.sub("<code>", code, prompt)


def code2docstring(code):
    prompt = get_code2docstring_template(code)
    code = iteratively_request_code(prompt, temperature=0.2, max_tokens=256, presence_penalty=0.2,
                                    frequency_penalty=0.3, stop=['"""', '/*', '\n\n\n'])
    return code


def get_oneliner_template(function_code, language):
    prompt = '''<comment start> Convert this function in <language> to a one line function<comment end>
    <function header>
        <code>

    <comment start> <language> one line version<comment end>:
    <function header>
        <return statement>'''
    prompt = inspect.cleandoc(prompt)
    prompt = re.sub("<comment start>", get_comment(language)[0], prompt)
    prompt = re.sub("<comment end>", get_comment(language)[1], prompt)
    prompt = re.sub("<language>", language, prompt)
    prompt = re.sub("<function header>", function_code.split('\n')[0], prompt)
    prompt = re.sub(
        "<code>", function_code[function_code.find('\n')+1:], prompt)
    prompt = re.sub("<return statement>", 'return', prompt)

    return prompt


def get_oneliner(function_code, language):
    prompt = get_oneliner_template(function_code, language)
    code = iteratively_request_code(prompt, temperature=0.2, max_tokens=150,
                                    frequency_penalty=0.5, stop=['"""', '\n'])
    return code


def get_unit_tests_template(function, language, context=''):
    parts = function.strip().split('\n')
    header = parts[0]
    body = '\n'.join(parts[1:])
    # get name of function i.e. the word that comes before a '('
    name = header.split('(')[0].split(' ')[-1]
    if context != '':
        template = context + "\n\n"
        template += f'{header}\n{body}\n\n'
    else:
        template = f'{header}\n{body}\n\n'
    cmt_start, cmt_end = get_comment(language)
    template += f'{cmt_start} Write a set of unit tests for the function {name} {cmt_end}\n\n'
    sl_cmt = get_comment(language, False)
    template += f'{sl_cmt} Unit Tests'
    return template


def code2ut(function, language, context):
    if 'python' in language.lower():
        prompt = "Python 3\n"
    else:
        prompt = f"{language}\n"
    prompt += get_unit_tests_template(function, language, context)
    temperature = 0
    code = iteratively_request_code(prompt, max_tokens=256, frequency_penalty=0.1, presence_penalty=0.1,
                                    temperature=temperature, stop=['#', '//', '/*'])
    while code.strip() == '':
        temperature += .1
        code = iteratively_request_code(prompt, max_tokens=256, frequency_penalty=0.1,
                                        presence_penalty=0.1, temperature=temperature,
                                        stop=['#', '"""', '//', '/*'])
    return code


def get_code_completion_template(code, task='', context=''):
    code_parts = code.split()
    if '/**' in code_parts[0]:
        docstring = code_parts[0] + "\n"
        i = 1
        while '*/' not in code_parts[i]:
            docstring += code_parts[i]
            i += 1
        docstring += code_parts[i] + "\n"
        function_code = code_parts[i+1:]
        if context != '':
            template = context + "\n\n"
        else:
            template = ''
        template += '"""Complete the following function"""\n'
        template += docstring
        template += function_code
    else:
        if context != '':
            template = context + "\n\n"
        else:
            template = ''
        if '"""' in code_parts[1] or "'''" in code_parts[1]:
            template += '"""Complete the following function"""'
            template += code
        else:
            template += f'"""Complete the following piece of code for {task}"""\n'
            template += code
    return template


def complete_code(code, task='', context=''):
    prompt = get_code_completion_template(code, task, context)
    if task == '':
        print("WARNING: Task not provided. Resulting code may not be accurate")
        temperature = 0.6
    else:
        temperature = 0.2
    code = iteratively_request_code(prompt, temperature=temperature, max_tokens=256, frequency_penalty=0.8,
                                    presence_penalty=0.4, stop=['\n\n\n', '"""'], best_of=3)
    return code
