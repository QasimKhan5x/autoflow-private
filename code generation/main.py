from myrunner import inference


context = '''
# <mask0>
'''

with open('input.txt') as f:
    context = context + f.read() + "'''"
print('='*20)
print('Inference Result',inference(context))