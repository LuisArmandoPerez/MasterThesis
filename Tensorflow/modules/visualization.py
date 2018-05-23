import matplotlib.pyplot as plt


def plot_latent_space_mapping(input_latent_code, output_latent_code, input_decoded, output_decoded):
    print("In order to get sequential coloring of data, all of the inputs of the function need to be sorted.")
    x = input_latent_code[:, 0]
    y = input_latent_code[:, 1]

    # Create figure
    fig = plt.figure(figsize=(20, 10))
    # Create input latent map
    ax = plt.subplot(121)
    line = ax.scatter(x, y, c = range(len(x)), cmap = 'Blues')
    ax.grid()
    ax.set_aspect(1)
    ax.set_title('Input latent space')
    ax.set_xlabel('Z1')
    ax.set_ylabel('Z2')
    # Create output latent map
    ax2 = plt.subplot(122)
    ax2.grid()
    ax2.set_title('Output latent space')
    ax2.set_xlabel('Z1')
    ax2.set_ylabel('Z2')
    ax2.set_aspect(1)
    line2 = ax2.scatter(output_latent_code[:, 0], output_latent_code[:, 1], c = range(len(output_latent_code)), cmap = 'Reds')
    # Create the reconstruction subaxes
    subax = plt.axes([0.65, 0.65, 0.2, 0.2])
    subax.set_visible(False)
    subax2 = plt.axes([0.65, 0.60, 0.2, 0.2])
    subax2.set_visible(False)

    def hover(event):
        # if the mouse is over the scatter points
        if line.contains(event)[0]:
            # find out the index within the array from the event
            ind, = line.contains(event)[1]["ind"]
            # get the figure size
            w, h = fig.get_size_inches() * fig.dpi
            w_inch, h_inch = fig.get_size_inches()
            ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
            hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
            # if event occurs in the top or right quadrant of the figure,
            # change the annotation box position relative to mouse.
            #         subax.xybox = (xybox[0]*ws, xybox[1]*hs)
            # make annotation box visible

            # Show the data reconstruction from input space
            subax.clear()
            figure_coord = fig.transFigure.inverted().transform((event.x, event.y))
            subax.set_position([figure_coord[0], figure_coord[1], 0.2, 0.2])
            subax.plot(range(len(input_decoded[ind])), input_decoded[ind], 'b')
            subax.set_xticks([])
            subax.set_yticks([])
            subax.set_visible(True)
            # Show the data reconstruction from output space
            subax2.clear()
            display_ax2 = ax2.transData.transform((output_latent_code[ind, 0], output_latent_code[ind, 1]))
            figure_ax2 = fig.transFigure.inverted().transform(display_ax2)
            subax2.set_position([figure_ax2[0], figure_ax2[1], 0.2, 0.2])
            subax2.plot(range(len(output_decoded[ind])), output_decoded[ind], 'b')
            subax2.set_xticks([])
            subax2.set_yticks([])
            subax2.set_visible(True)
            ax2.plot(output_latent_code[ind, 0], output_latent_code[ind, 1], ls="", markersize=10, markeredgewidth=1.5,
                     marker="o", markeredgecolor="k", markerfacecolor='w', color="r")
        else:
            ax2.clear()
            ax2.grid()
            ax2.set_title('Output latent space')
            ax2.set_xlabel('Z1')
            ax2.set_ylabel('Z2')
            ax2.scatter(output_latent_code[:, 0], output_latent_code[:, 1],c = range(len(output_latent_code)), cmap = 'Reds')
            # if the mouse is not over a scatter point
            subax2.set_visible(False)
            subax.set_visible(False)
        fig.canvas.draw_idle()

    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover)
    plt.show()
