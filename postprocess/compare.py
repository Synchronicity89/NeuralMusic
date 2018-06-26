import mido
import numpy as np
from operator import itemgetter

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
        return (positives / len(similarity)) * 100

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

    def compareData(self, trackdata, recreated, file):
        allTracksCompared = self.compareRecreated(trackdata, recreated)
        allAverage = self.getTotalAverage(allTracksCompared)
        bestMatch = self.getBestMatch(allTracksCompared)
        worstMatch = self.getWorstMatch(allTracksCompared)
        top5 = self.getTop5(allTracksCompared)
        file.write('Total average similarity: ' + str(allAverage)+'\n')
        file.write('Best match: ' + str(bestMatch) + ' %\n')
        file.write('Worst match: ' + str(worstMatch) + ' %\n')
        note_information = self.findSimilarNotes(trackdata, recreated, file)
        note_information = sorted(note_information, key=itemgetter(1), reverse=True)
        not_present_in_input = self.findDifferentNotes(trackdata, recreated)
        not_present_in_input = sorted(not_present_in_input, key=itemgetter(1), reverse=True)
        self.printSimilarity(note_information, file)
        self.printDifference(not_present_in_input,  file)

    def createUniqueNotesList(self, trackdata, recreated):
        list = []
        for track in trackdata:
            indices = np.unique(np.nonzero(track)[1])
            list.append(indices)
        return list, np.unique(np.nonzero(recreated)[1])

    def findSimilarNotes(self, trackdata, recreated, file):
        output = []
        mostVariatedSolo = 0
        leastVariatedSolo = 100
        list, generated_solo = self.createUniqueNotesList(trackdata, recreated)
        file.write('Number of unique notes in generated solo: ' + str(len(generated_solo))+'\n')
        for solo in list:
            if len(solo) > mostVariatedSolo:
                mostVariatedSolo = len(solo)
            if len(solo) < leastVariatedSolo:
                leastVariatedSolo = len(solo)
            difference = np.setdiff1d(solo, generated_solo)
            persentage = (len(solo)-len(difference))/len(solo)
            tuple = difference, persentage
            output.append(tuple)
        file.write('The most variated solo has ' + str(mostVariatedSolo) + ' unique notes\n')
        file.write('The least variated solo has ' + str(leastVariatedSolo) + ' unique notes\n')
        return output

    def findDifferentNotes(self, trackdata, recreated):
        output = []
        list, generated_solo = self.createUniqueNotesList(trackdata, recreated)
        for solo in list:
            difference = np.setdiff1d(generated_solo, solo)
            persentage = (len(difference))/len(generated_solo)
            tuple = difference, persentage
            output.append(tuple)
        return output

    def printSimilarity(self, note_information, file):
        hundred = 0
        ninety = 0
        eighty = 0
        seventy = 0
        sixty = 0
        fifty = 0
        forty = 0
        thirty = 0
        twenty = 0
        ten = 0
        belowTen = 0

        for solo in note_information:
            persentage = int(solo[1] * 100)
            if persentage == 100:
                hundred = hundred + 1
            elif persentage in range(90, 99):
                ninety = ninety + 1
            elif persentage in range(80, 89):
                eighty = eighty + 1
            elif persentage in range(70, 79):
                seventy = seventy + 1
            elif persentage in range(60, 69):
                sixty = sixty + 1
            elif persentage in range(50, 59):
                fifty = fifty + 1
            elif persentage in range(40, 49):
                forty = forty + 1
            elif persentage in range(30, 39):
                thirty = thirty + 1
            elif persentage in range(20, 29):
                twenty = twenty + 1
            elif persentage in range(10, 19):
                ten = ten + 1
            else:
                belowTen = belowTen + 1

        file.write('Number of solos that contains the same notes as the generated solo\n')
        file.write('100 %: ' + str(hundred)+'\n')
        file.write('90 - 99%: ' + str(ninety)+'\n')
        file.write('80 - 89%: ' + str(eighty)+'\n')
        file.write('70 - 79%: ' + str(seventy)+'\n')
        file.write('60 - 69%: ' + str(sixty)+'\n')
        file.write('50 - 59%: ' + str(fifty)+'\n')
        file.write('40 - 49%: ' + str(forty)+'\n')
        file.write('30 - 39%: ' + str(thirty)+'\n')
        file.write('20 - 29%: ' + str(twenty)+'\n')
        file.write('10 - 19%: ' + str(ten)+'\n')
        file.write('< 10 %: ' + str(belowTen)+'\n')

    def printDifference(self, note_information, file):
        hundred = 0
        ninety = 0
        eighty = 0
        seventy = 0
        sixty = 0
        fifty = 0
        forty = 0
        thirty = 0
        twenty = 0
        ten = 0
        belowTen = 0
        zero = 0

        for solo in note_information:
            persentage = int(solo[1] * 100)
            if persentage == 100:
                hundred = hundred + 1
            elif persentage in range(90, 99):
                ninety = ninety + 1
            elif persentage in range(80, 89):
                eighty = eighty + 1
            elif persentage in range(70, 79):
                seventy = seventy + 1
            elif persentage in range(60, 69):
                sixty = sixty + 1
            elif persentage in range(50, 59):
                fifty = fifty + 1
            elif persentage in range(40, 49):
                forty = forty + 1
            elif persentage in range(30, 39):
                thirty = thirty + 1
            elif persentage in range(20, 29):
                twenty = twenty + 1
            elif persentage in range(10, 19):
                ten = ten + 1
            elif persentage in range(1, 9):
                belowTen = belowTen + 1
            else:
                zero = zero + 1

        file.write('Difference between generated solo and input solo. If 100 percent, no notes in the generated solo are present in the input\n')
        file.write('100 %: ' + str(hundred)+'\n')
        file.write('90 - 99%: ' + str(ninety)+'\n')
        file.write('80 - 89%: ' + str(eighty)+'\n')
        file.write('70 - 79%: ' + str(seventy)+'\n')
        file.write('60 - 69%: ' + str(sixty)+'\n')
        file.write('50 - 59%: ' + str(fifty)+'\n')
        file.write('40 - 49%: ' + str(forty)+'\n')
        file.write('30 - 39%: ' + str(thirty)+'\n')
        file.write('20 - 29%: ' + str(twenty)+'\n')
        file.write('10 - 19%: ' + str(ten)+'\n')
        file.write('1 - 9 %: ' + str(belowTen)+'\n')
        file.write('0 %: ' + str(zero)+'\n\n')
