import sys

def error_message_details(error, error_details: sys):
    """
    This function provides a more structured error message including the filename and line number
    where the error occurred.

    Args:
        error (str): The error message or exception raised.
        error_details (sys): The sys module, used to extract detailed error information.

    Returns:
        str: A structured error message that includes the script name, line number, and the error message.

    Raises:
        CustomException: If any exception or error occurs.
    """
    
    # Extract the traceback information from the exception
    _, _, exc_tb = error_details.exc_info()
    
    # Get the filename where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Get the line number where the error occurred
    line_no = exc_tb.tb_lineno
    
    # Create a structured error message
    error_message = f"Error occurred in python script [{file_name}] at line number [{line_no}] with error message [{str(error)}]"
    
    return error_message

class CustomException(Exception):
    """
    A custom exception class that formats and provides detailed error messages.

    Attributes:
        error_message (str): The detailed error message returned by error_message_details function.
        error_details (sys): The sys module, used to extract traceback information.

    Methods:
        __str__(): Returns the formatted error message when the exception is raised.
    """
    
    def __init__(self, error_message, error_details: sys):
        """
        Initializes the CustomException with the provided error message and details.

        Args:
            error_message (str): The error message or exception raised.
            error_details (sys): The sys module to extract error details.
        """
        # Initialize the base Exception class with the error message
        super().__init__(error_message)
        
        # Format the error message using the error_message_details function
        self.error_message = error_message_details(error_message, error_details)
        
    def __str__(self):
        """Override the default __str__ method to return the formatted error message."""
        return self.error_message
