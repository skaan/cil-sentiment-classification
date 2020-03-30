# informal interface for preprocessing steps

class PreprocessingInterface:

    # methods we want each step to have
    def set_paths(self, source, destination):
        self.source = source
        self.destination = destination

    def run(self):
        """run preprocessing step"""

        # Code you want to execute at each step before run
        if self.source is None:
            raise Exception("Source not set", "You haven't set the input source.")
        else if self.destination is None:
            raise Exception("Destination not set", "You haven't set the output destination.")

        pass
