def run_action_with_recovery(action):
    """""\"
Summary of run_action_with_recovery.

Parameters
----------
action : type
    Description of parameter `action`.

Returns
-------
None
""\""""
    try:
        action()
    except Exception as e:
        print(f'Error: {e}')
        rollback_to_last_stable()


def rollback_to_last_stable():
    """""\"
Summary of rollback_to_last_stable.


Returns
-------
None
""\""""
    print('Rolling back to the last stable version.')
