import mido
import numpy as np

class Compare:
    def __init__(self):
        super(Compare, self).__init__()

    def compareRecreated(elf, trackdata, recreated):
        allTracksCompared = []
        for track in trackdata:
            similarity = []
            for i, timestep in enumerate(track):
                if i < len(recreated):
                    x = recreated[i]
                    if (np.array_equal(timestep, x)):
                        similarity.append(True)
                    else:
                        similarity.append(False)
            allTracksCompared.append(similarity)
        return allTracksCompared

    def getPersentageSimilarity(self, similarity):
        positives = 0
        for elem in similarity:
            if elem is True:
                positives = positives + 1
        return positives / len(similarity)

    def getTotalAverage(self, allTracksCompared):
        avg = 0
        for track in allTracksCompared:
            avg = avg + self.getPersentageSimilarity(track)
        return avg / len(allTracksCompared)

    def getPersentages(self, allTracksCompared):
        similarities = []
        for track in allTracksCompared:
            similarities.append(self.getPersentageSimilarity(track))
        return similarities

    def getBestMatch(self, allTracksCompared):
        best = 0
        for track in allTracksCompared:
            avg = self.getPersentageSimilarity(track)
            if best < avg:
                best = avg
        return best

    def getWorstMatch(self, allTracksCompared):
        worst = 100
        for track in allTracksCompared:
            avg = self.getPersentageSimilarity(track)
            if worst > avg:
                worst = avg
        return worst

    def getTop5(self, allTracksCompared):
        persentages = self.getPersentages(allTracksCompared)
        persentages.sort(reverse=True)
        return persentages[:5]

    def compareData(self, trackdata, recreated):
        allTracksCompared = self.compareRecreated(trackdata, recreated)
        allAverage = self.getTotalAverage(allTracksCompared)
        bestMatch = self.getBestMatch(allTracksCompared)
        worstMatch = self.getWorstMatch(allTracksCompared)
        top5 = self.getTop5(allTracksCompared)
        print('Total average similarity: ' + str(allAverage))
        print('Best match: ' + str(bestMatch) + ' %')
        print('Worst match: ' + str(worstMatch) + ' %')
        print('Top 5 similar solos ')
        print(top5)
