import sys

def error_message_detail(error,error_detail:sys):
    _,_,exc_tab=error_detail.exc_info()
    file_name=exc_tab.tb_frame.f_code.co_filename

    error_detail="Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tab.tb_lineno,str(error)
    )

    return error_detail


class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_detail)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message
    
