from nltk.tokenize import TweetTokenizer

class Personality:
    """ Class that handles the personality traits of a person.

    Personality traits following the Big Five model.
    """

    def __init__(self, extrovert, stable, agreeable, conscientious, open_trait):
        """ Constructor for the Personality class

        Args:
            extrovert (float): value for the extrovert trait.
            stable (float):  value for the stable trait.
            agreeable (float):  value for the agreeable trait.
            conscientious (float):  value for the conscientious trait.
            open_trait (float):  value for the open trait.
        """
        self._extroverted = extrovert
        self._stable = stable
        self._agreeable = agreeable
        self._conscientious = conscientious
        self._open = open_trait

    @property
    def extroverted(self):
        return self._extroverted

    @property
    def stable(self):
        return self._stable

    @property
    def agreeable(self):
        return self._agreeable

    @property
    def conscientious(self):
        return self._conscientious

    @property
    def open(self):
        return self._open

    def __repr__(self):
        format_string = "E: {:2.4}, S: {:<4.2}, A: {:<3.2}, C: {:<.4}, O: {:2.4}"
        return format_string.format(self.extroverted, self.stable,
                                    self.agreeable, self.conscientious,
                                    self.conscientious)


class Author:
    """ Class that encapsulates a single sample """

    def __init__(self, author_id, gender, age_range, personality_traits):
        """

        Args:
            gender (str):
            age_range (str):
            personality_traits (Personality):
        """
        self._author_id = author_id
        self._tweets = []
        self._gender = gender
        self._age = age_range
        self._personality = personality_traits
        self._padded = []

    @property
    def author_id(self):
        return self._author_id

    @property
    def tweets(self):
        return self._tweets

    @property
    def gender(self):
        return self._gender

    @property
    def age(self):
        return self._age

    @property
    def personality(self):
        return self._personality

    @property
    def padded(self):
        return self._padded

    def add(self, tweet):
        """ Add a tweet to the collections of tweets writen by this author.
        Args:
            tweet(str): a tweet writen by this author.

        Returns:
            Modify the list of tweets written by an author.

        Todo:
            Perform some kind of preprocess of the data
        """
        self._tweets.append(tweet.strip())

    def num_tweets(self):
        """ Return the number of tweets that we have for each author """
        return len(self._tweets)

    def tokenize(self):
        """ Tokenize each one of the authors tweets. """
        # reduce_len: Replace repeated character sequences of
        # length 3 or greater with sequences of length 3.
        # strip_handles: Remove Twitter username handles from text.
        # TODO: Add the Stanford Tokenizer ?
        # TODO: Tokenize the urls
        # In this dataset usernames are already annonimized, therefore,
        # there is no need to clean them.
        tknzr = TweetTokenizer(reduce_len=True, preserve_case=True)
        for pos in range(len(self._tweets)):
            self._tweets[pos] = tknzr.tokenize(self._tweets[pos])

    def max_words(self):
        """ Return the maximum number of words that an author have used """
        # If the author has no tweets, then the length is 0
        if not self._tweets:
            return 0
        else:
            num_words_list = []
            for tweet in self._tweets:
                # If it has already tokenized only count the number of words,
                # otherwise, tokenize and count.
                if isinstance(tweet,list):
                    num_words_list.append(len(tweet))
                else:
                    # TODO: Try to refactor this
                    tknzr = TweetTokenizer(reduce_len=True, preserve_case=True)
                    num_words_list.append(len(tknzr.tokenize(tweet)))
            return max(num_words_list)

    def pad(self, max_sequence):
        """ Pad all the tweets in order to have sequences of same length

        Args:
            max_sequence (int): max number of words in a tweet from
                the training dataset

        Returns:

        """
        # If the tweets are not already tokenize, do it.
        if self._tweets and isinstance(self._tweets[0], str):
            self.tokenize()

        if not self._padded:
            for tweet in self._tweets:
                len_padding = max_sequence - len(tweet)
                padded_tweet = tweet + ['<pad>'] * len_padding
                # TODO: move it to unittest
                assert len(padded_tweet) == max_sequence
                self._padded.append(padded_tweet)

            # TODO: move it to unittest
            assert len(self._tweets) == len(self._padded)

    def __repr__(self):
        author_str = "* #{0:<10} [{4}]\n  - Age: {1:<5}, Gender: {2:<2}\n  - {3}"
        return author_str.format(self._author_id[:10], self._age, self._gender,
                                 self._personality.__repr__(), self.num_tweets())


class Corpus:
    def __init__(self):
        self._samples = []

    @property
    def samples(self):
        return self._samples

    def __len__(self):
        return len(self._samples)

    def add(self, author_sample):
        """ Add a new sample to the dataset

        Args:
            author_sample: a sample of an author

        Returns:
            If the author_sample is an instance of the Author class
            modify the list  of samples adding this new sample

        """
        if isinstance(author_sample, Author):
            self._samples.append(author_sample)
        else:
            reason_string = "The new sample trying to add is not an instance of the Author class."
            current_type_string = " The object is of class: {}".format(type(author_sample))
            raise TypeError(reason=reason_string+current_type_string)

    def min(self):
        """ Return the minimum number of samples for an author """
        return min([sample.num_tweets() for sample in self._samples])

    def max(self):
        """ Return the maximum number of samples for an author """
        return max([sample.num_tweets() for sample in self._samples])

    def num_tweets(self):
        return [sample.num_tweets() for sample in self._samples]

    def pad(self, max_sequence=None):
        """
        Pad each tweet to match the longest text sample present in the dataset.

        Args:
            max_sequence (int): If we are padding the training dataset,
                                it would be None and the method will return
                                the maximum number of words.
                                If we are padding the test dataset,
                                this value contains the maximum number of words
                                seen in the training dataset.

        Returns: max number of words in the dataset, and modify the samples
            such as each text sample has the same size.

        """
        if not max_sequence:
            max_sequence = max([author.max_words() for author in self._samples])
        for author in self._samples:
            author.pad(max_sequence)

        return max_sequence