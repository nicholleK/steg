# formats the output of a pipeline's classification result on an image

class ClassificationResult:

    def __init__(self,
                predictedClass: str,
                confidenceScore: float,
                allProbabilities: dict,
                threshold: float=0.60,
                isFlagged: bool= None

                 ):

        self.predictedClass = predictedClass
        self.confidenceScore = round(confidenceScore, 4)
        self.allProbabilities = allProbabilities
        self.threshold = threshold
        self.isFlagged = isFlagged if isFlagged is not None else (confidenceScore < threshold) # flag if confidence is below threshold

        # Methods

    # manually marks a result as flagged
    def flag(self):
        self.isFlagged = True

    def getMetrics(self) -> dict:
         # returns a dictionary of key result values
        return {
            "predictedClass": self.predictedClass,
            "confidenceScore": self.confidenceScore,
            "allProbabilities": self.allProbabilities,
            "isFlagged": self.isFlagged,
        }

    def toReport(self) -> str:
        # returns multi-line formatted report string

        flag_status = "YES. Manual review recommended" if self.isFlagged else "No manual review recommended"

        prob_lines = "\n".join(
            f"{cls:<12}: {prob:4f} ({prob:.1%})"
            for cls, prob in self.allProbabilities.items()
        )

        report = (
            f"CLASSIFICATION RESULT\n"
            f"{'-' * 40}\n"
            f"Predicted Class: {self.predictedClass}\n"
            f"Confidence Score: {self.confidenceScore:.4f}\n"
            f"Flagged: {flag_status}\n"
            f"Threshold: {self.threshold:.0%}\n"
            f"\nCLASS PROBABILITIES\n"
            f"{'-' * 40}\n"
            f"{prob_lines}\n"
        )

        return report

    def __repr__(self):
        return (f"ClassificationResult(class={self.predictedClass},"
                f" confidenceScore={self.confidenceScore:.4f}),"
                f"flagged: {self.isFlagged}")
