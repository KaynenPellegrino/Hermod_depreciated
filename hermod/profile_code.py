# hermod/profile_code.py

"""
Module: profile_code.py

Profiles the Hermod application to analyze performance.
"""

import cProfile
import pstats
import io
from hermod.main import main
from hermod.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()

def profile_application():
    """
    Profiles the Hermod application to analyze performance.
    """
    logger.info("Starting performance profiling using cProfile...")
    pr = cProfile.Profile()
    pr.enable()

    # Run the main function of Hermod
    main()

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    # Save the profiling report to a file
    report_file = 'logs/performance_profile.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(s.getvalue())
    logger.info(f"Performance profiling report saved to {report_file}")

if __name__ == "__main__":
    profile_application()
