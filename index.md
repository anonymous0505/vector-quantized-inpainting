---
layout: default
---

This is the companion website for the paper
Interactive Generation of Spectrograms with Vector-Quantized Inpainting.

# Introduction

In this paper, we adapt the [VQ-VAE-2 architecture][vq-vae-2] to the interactive generation of sounds.

To this end, we introduce a modified version of the Mel-IF representation proposed in the
[GANSynth paper][gansynth], that allows to remove irrelevant noise in this representation,
making it more adapted to quantization via the VQ-VAE-2.

We then follow the approach proposed in the original VQ-VAE-2 paper and generate new sounds by sampling
from the distribution of codemaps produced the trained VQ-VAE-2.
This has the advantage of turning the intractable autoregressive modeling of full-scaled \\( 1024 \times 128 \\) ,
real-valued spectrograms into just sampling two small discrete codemaps of sizes
\\( 32 \times 4 \\) and \\( 64 \times 8 \\).

## VQ-VAE-2 architecture

The VQ-VAE-2 is a two-layer hierarchical model based on the common VAE encoder-decoder framework and
was originally introduced for the generation of images. This model learns to encode
2D images onto two, 2D integer arrays of query indexes: the top codemap and the bottom codemap.

This quantization to integers is performed by learning within each of the two quantized layers a dictionary of code vectors
onto which the latent encoder representations are projected.
These two codemaps are conditionally bound. Indeed, the encoder of the VQ-VAE-2 first downsamples
and quantizes the input image into the bottom map. This first representation is then further
downsampled and again quantized, yielding the top codemap.

In the decoder, the top codemap is first reconstructed using the code vectors of the learned dictionary.
This reconstruction is then used as a conditioning information for the decoding of the bottom codemap.

This architecture is represented in the following diagram.

![Placeholder for vqvae diagram image](/assets/images/vqvae-simple.png)

## Image-like representation for audio

In order to directly apply the VQ-VAE-2 architecture to sounds, we introduce a modified
version of the Mel-IF representation used in the GANSynth model, which helps the VQ-VAE-2
learn more robust codes by removing irrelevant noise in the phase at low sound amplitude.
We furthermore tweak the Mel-scale to make it "more logarithmic" by lowering the _break frequency_,
resulting in even more resolution in the lower frequencies.

We take the following sound sample as example (sampled from the NSynth validation set).

(/assets/audio/sample.wav)

On the following figure, we display the original, GANSynth-style Mel-IF representation for this sound,
without phase thresholding and mel-scale compression. We also show the associated top and bottom codemaps
obtained from a model trained on this representation.

![Placeholder for mel-IF representation image](/assets/images/mel_IF-original_gansynth.png)

We now show the same spectrogram using our modified representation and the associated codemaps,
obtained from a VQ-VAE-2 trained on these representations.

![Placeholder for mel-IF representation image](/assets/images/mel_IF-ours.png)

## Sequence-masked Transformers for inpainting

Sampling new sounds can now be done dy directly sampling from the joint probability of top and bottom codemaps:
\\[ p(c^{T}, c^{B}) \\]
This probability can be readily factorized using the classic Bayes-rule
into the following product of conditional probabilities \\[ p(c^{T}, c^{B}) = p(c^{T}) * p(c^{B} | c^{T}) \\]

We model these two probabilities using two distinct autoregressive Transformers.
Since we also want to perform inpaiting on the generated sounds,
that is, resample single tokens in the codemaps for interactive generation,
the chosen modeling must account for all past and future context of each token.

We furthermore use the strong alignment present between the top and bottom codemap
to remove some conditional dependencies in the factorized expression of \\( p(c^{B} | c^{T}) \\). Indeed, since the bottom map is an upsampled version of the bottom map, we model the conditional dependency from top to bottom in the patch-based fashion resulting from this upsampling scheme. We consider that a token \\( i \\) is conditionally depends only on a single token \\( k(i) \\) from which it was upsampled.

We therefore model the following two factorized families of probabilities:
\\[
p(c^{T}) = \prod_{i}{ p(c^{T}\_i | c^{T}\_{\<i}, c^{T}\_{\>i}) }
\\]

and

\\[
p(c^{B} \vert c^{T}) = \prod_{i}{ p(c^{B}\_i | c^{B}\_{\<i}, c^{T}\_{k(i)}) }
\\]

The complete inpainting process is represented in this diagram:

![Placeholder for proposed approach diagram](/assets/images/inpainting_process.png)

On the right-hand side of this diagram we can see the inpainting procedure in action:

1. The codemaps extracted by the VQ-VAE-2 encoder are linearized and fed to the Transformers (see paper for details on the linearization shceme),
2. The token circled in pink from the top map is resampled using the conditional probability computed by the top Transformer,
3. The underlying tokens in the aligned bottom map are reseampled according to the conditional probabilities computed by the bottom Transformers.
4. The resulting, inpainted codemaps are decoded back to a spectrogram using the VQ-VAE-2 decoder.

# Experiments

We now present audio samples for the different components of our approach.

The models were trained on the NSynth dataset, using a custom 80/20 train/valid split since the original was made across instrument types, which is not relevant for our approach.

## VQ-VAE reconstructions

For this first experiment, we only showcase the trained VQ-VAE's reconstruction ability.

The experimental setup is as follows: we sample 16 random examples from the NSynth validation split and compute their reconstruction using the VQ-VAE-2 encoder.

<div markdown="0">
<table class="tableFixHead">
<colgroup>
<col/>
<col/>
<col/>
</colgroup>
<thead>
<tr class="header">
<th>Instrument</th>
<th>Pitch</th>
<th>Audio</th>
<th></th>
</tr>
</thead>
<tbody style="overflow-y: scroll; height: 300px">

{% assign formats = "ogg, flac, wav" | split: ", " %}
{% assign sample_categories = "original, reconstruction" | split: ", " %}

{% for sample in site.data.reconstructions %}
    <tr>
    <td> {{ sample[1].instrument_family_str | capitalize}} </td>
    <td> {{ sample[1].pitch }} </td>
    <td>
        <table>
            <tbody>
            {% for sample_category in sample_categories %}
            <tr>
                <td> {{ sample_category | capitalize | truncate: 5, "." }} </td>
                <td>
                    <audio controls preload="none">
                        {% for format in formats %}
                            <source src="{{ site.baseurl }}assets/audio/reconstructions/{{ sample[0] }}-{{sample_category}}.{{ format }}"
                                alt="sample" type="audio/{{ format }}">
                        {% endfor %}
                    </audio>
                </td>
            </tr>
            {% endfor %}
            </tbody>
        </table>
    </td>

    </tr>
{% endfor %}
</tbody>
</table>
</div>

[vq-vae-2]: https://arxiv.org/abs/1906.00446 "ArXiV: VQ-VAE-2"
[gansynth]: https://arxiv.org/abs/1902.08710 "ArXiV: GANSynth"