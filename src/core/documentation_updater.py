def update_readme(content):
    """""\"
Summary of update_readme.

Parameters
----------
content : type
    Description of parameter `content`.

Returns
-------
None
""\""""
    with open('README.md', 'a') as readme:
        readme.write('\n' + content)
    print('README.md updated.')
